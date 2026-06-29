# fuzzytool desktop (Tauri + Python sidecar)

A small **Tauri** desktop app that drives the `fuzzytool` library through a
**Python sidecar**. The Rust core spawns the sidecar and relays a line-delimited
JSON protocol; the web frontend renders sliders and a live control surface.

```
desktop/
  index.html, src/main.ts      frontend (Vite, vanilla TS): sliders + heatmap
  sidecar/fuzzy_server.py       Python sidecar: fuzzytool behind a JSON stdin/stdout loop
  src-tauri/                    Rust backend: spawns the sidecar, exposes `fuzzy_call`
```

The sidecar exposes the bundled `datasets.credit_risk` Mamdani system. All the
fuzzy work (inference, fired-rule explanation via
`integrations.agents.explain`, control surface) is done by fuzzytool — the app
is just a shell.

## Architecture

```
 frontend (webview)  ──invoke("fuzzy_call", json)──▶  Rust (Tauri)
                                                        │  stdin/stdout (JSON lines)
                                                        ▼
                                              python sidecar (fuzzytool)
```

Protocol (one JSON object per line):

| action | request | result |
|--------|---------|--------|
| `describe` | `{"action":"describe"}` | input/output variables, ranges, terms |
| `infer` | `{"action":"infer","inputs":{"score":700,"dti":20}}` | `{output, fired_rules}` |
| `surface` | `{"action":"surface","x":"score","y":"dti","n":40}` | grid `{x,y,z}` |

## Prerequisites

- Node ≥ 18 and Rust (stable) with the [Tauri v2 prerequisites](https://tauri.app/start/prerequisites/).
- A Python with `fuzzytool` installed (`pip install fuzzytool`).

## Run (development)

```bash
cd desktop
npm install

# Point the sidecar at a Python that has fuzzytool installed (defaults to python3):
export FUZZY_SIDECAR_PYTHON=/path/to/venv/bin/python

npm run tauri dev
```

The Rust backend runs `$FUZZY_SIDECAR_PYTHON sidecar/fuzzy_server.py`. Override
`FUZZY_SIDECAR_SCRIPT` to use a different script.

## Build a distributable (frozen sidecar)

For a self-contained app, freeze the sidecar so end users need no Python:

```bash
pip install pyinstaller
pyinstaller --onefile --name fuzzy_server \
    --collect-submodules fuzzytool \
    desktop/sidecar/fuzzy_server.py        # -> dist/fuzzy_server

# Ship the binary and point the app at it (no script argument):
export FUZZY_SIDECAR_PYTHON=/path/to/fuzzy_server
export FUZZY_SIDECAR_SCRIPT=

cd desktop
npm run tauri icon src-tauri/icons/icon.png   # generate platform icons once
npm run tauri build
```

For a production bundle you would register `fuzzy_server` as a Tauri
[sidecar binary](https://tauri.app/develop/sidecar/) (named with the target
triple under `src-tauri/binaries/`) and resolve it via the shell plugin instead
of an env var; the env-var path above keeps this prototype simple.

## Test the sidecar without the GUI

```bash
printf '%s\n' \
  '{"id":1,"action":"describe"}' \
  '{"id":2,"action":"infer","inputs":{"score":800,"dti":10}}' \
  | python desktop/sidecar/fuzzy_server.py
```
