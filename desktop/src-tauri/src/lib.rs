//! Tauri backend: spawns the fuzzytool Python sidecar and relays JSON to it.
//!
//! The sidecar speaks a line-delimited JSON protocol on stdin/stdout (see
//! `sidecar/fuzzy_server.py`). Requests and responses are 1:1 and ordered, so a
//! single mutex-guarded write-then-read is enough.
//!
//! The sidecar command is resolved from environment variables, falling back to
//! running the script with `python3` (handy for development). For distribution,
//! freeze the sidecar with PyInstaller and point `FUZZY_SIDECAR_PYTHON` at the
//! resulting binary (with no script argument).

use std::io::{BufRead, BufReader, BufWriter, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Mutex;

use tauri::State;

struct Sidecar {
    #[allow(dead_code)] // kept alive so the child process is not reaped
    child: Child,
    writer: BufWriter<ChildStdin>,
    reader: BufReader<ChildStdout>,
}

impl Sidecar {
    fn spawn() -> std::io::Result<Self> {
        let python =
            std::env::var("FUZZY_SIDECAR_PYTHON").unwrap_or_else(|_| "python3".to_string());
        let script = std::env::var("FUZZY_SIDECAR_SCRIPT").unwrap_or_else(|_| {
            concat!(env!("CARGO_MANIFEST_DIR"), "/../sidecar/fuzzy_server.py").to_string()
        });

        let mut command = Command::new(python);
        // An empty script var means `python` is itself the frozen sidecar binary.
        if !script.is_empty() {
            command.arg(script);
        }
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn()?;

        let writer = BufWriter::new(child.stdin.take().expect("sidecar stdin"));
        let reader = BufReader::new(child.stdout.take().expect("sidecar stdout"));
        Ok(Self { child, writer, reader })
    }

    fn call(&mut self, request: &str) -> std::io::Result<String> {
        self.writer.write_all(request.as_bytes())?;
        self.writer.write_all(b"\n")?;
        self.writer.flush()?;

        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::BrokenPipe,
                "sidecar closed the connection",
            ));
        }
        Ok(line)
    }
}

/// Relay one JSON request to the sidecar and return its JSON response line.
#[tauri::command]
fn fuzzy_call(request: String, state: State<'_, Mutex<Sidecar>>) -> Result<String, String> {
    let mut sidecar = state.lock().map_err(|e| e.to_string())?;
    sidecar.call(&request).map_err(|e| e.to_string())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let sidecar = Sidecar::spawn().expect("failed to start the fuzzytool sidecar");
    tauri::Builder::default()
        .manage(Mutex::new(sidecar))
        .invoke_handler(tauri::generate_handler![fuzzy_call])
        .run(tauri::generate_context!())
        .expect("error while running the fuzzytool desktop app");
}
