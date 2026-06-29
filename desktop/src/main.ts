import { invoke } from "@tauri-apps/api/core";

// --- talk to the Python sidecar through the Rust command --------------------

type Resp<T> = { id?: number; ok: boolean; result?: T; error?: string };

async function fuzzy<T>(req: object): Promise<T> {
  const raw = await invoke<string>("fuzzy_call", { request: JSON.stringify(req) });
  const resp = JSON.parse(raw) as Resp<T>;
  if (!resp.ok) throw new Error(resp.error ?? "sidecar error");
  return resp.result as T;
}

interface VarInfo { name: string; low: number; high: number; terms: string[]; }
interface Describe { inputs: VarInfo[]; output: VarInfo; }
interface Infer { output: number; fired_rules: { rule: string; firing: number }[]; }
interface Surface { x: number[]; y: number[]; z: number[][]; x_name: string; y_name: string; }

const state: Record<string, number> = {};
let desc: Describe;
let surface: Surface | null = null;

// --- surface heatmap --------------------------------------------------------

function colour(t: number): string {
  // viridis-ish: dark blue -> green -> yellow
  const r = Math.round(255 * Math.min(1, Math.max(0, 1.4 * t - 0.4)));
  const g = Math.round(255 * Math.min(1, Math.max(0, 1.1 * t)));
  const b = Math.round(255 * Math.min(1, Math.max(0, 0.8 - 0.9 * t)));
  return `rgb(${r},${g},${b})`;
}

function drawSurface() {
  const canvas = document.getElementById("surface") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  if (!surface) return;
  const { x, y, z } = surface;
  const nx = x.length, ny = y.length;
  let lo = Infinity, hi = -Infinity;
  for (const row of z) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const cw = canvas.width / nx, ch = canvas.height / ny;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      ctx.fillStyle = colour((z[j][i] - lo) / (hi - lo || 1));
      // y grows upward
      ctx.fillRect(i * cw, canvas.height - (j + 1) * ch, cw + 1, ch + 1);
    }
  }
  // marker for the current operating point
  const xv = desc.inputs[0], yv = desc.inputs[1];
  const px = ((state[xv.name] - xv.low) / (xv.high - xv.low)) * canvas.width;
  const py = canvas.height - ((state[yv.name] - yv.low) / (yv.high - yv.low)) * canvas.height;
  ctx.strokeStyle = "white"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke();
}

// --- inference + UI ---------------------------------------------------------

async function refresh() {
  const res = await fuzzy<Infer>({ action: "infer", inputs: state });
  document.getElementById("output")!.textContent =
    `${desc.output.name} = ${res.output.toFixed(2)}`;
  document.getElementById("rules")!.innerHTML = res.fired_rules.length
    ? res.fired_rules.map(r => `<div><code>${r.rule}</code> &middot; ${r.firing}</div>`).join("")
    : "<em>no rules fired</em>";
  drawSurface();
}

function buildControls() {
  const root = document.getElementById("controls")!;
  for (const v of desc.inputs) {
    state[v.name] = (v.low + v.high) / 2;
    const wrap = document.createElement("div");
    wrap.className = "control";
    const label = document.createElement("label");
    const span = document.createElement("span");
    label.textContent = v.name;
    span.textContent = state[v.name].toFixed(1);
    label.appendChild(span);
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(v.low); slider.max = String(v.high);
    slider.step = String((v.high - v.low) / 200);
    slider.value = String(state[v.name]);
    slider.addEventListener("input", () => {
      state[v.name] = Number(slider.value);
      span.textContent = state[v.name].toFixed(1);
      refresh();
    });
    wrap.appendChild(label); wrap.appendChild(slider);
    root.appendChild(wrap);
  }
}

async function main() {
  desc = await fuzzy<Describe>({ action: "describe" });
  buildControls();
  surface = await fuzzy<Surface>({
    action: "surface", x: desc.inputs[0].name, y: desc.inputs[1].name, n: 40,
  });
  document.getElementById("xlabel")!.textContent = `x: ${surface.x_name}`;
  document.getElementById("ylabel")!.textContent = `y: ${surface.y_name}`;
  await refresh();
}

main().catch(err => {
  document.getElementById("output")!.textContent = `error: ${err}`;
});
