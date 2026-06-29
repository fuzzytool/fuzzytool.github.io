import { invoke } from "@tauri-apps/api/core";
import { openEditor, type Spec } from "./editor";

// --- sidecar bridge ---------------------------------------------------------

type Resp<T> = { ok: boolean; result?: T; error?: string };

async function fuzzy<T>(req: object): Promise<T> {
  const raw = await invoke<string>("fuzzy_call", { request: JSON.stringify(req) });
  const resp = JSON.parse(raw) as Resp<T>;
  if (!resp.ok) throw new Error(resp.error ?? "sidecar error");
  return resp.result as T;
}

interface VarInfo { name: string; low: number; high: number; terms: string[]; }
interface Describe { name: string; defuzz: string; inputs: VarInfo[]; output: VarInfo; rules: string[]; }
interface SystemsInfo { systems: { id: string; name: string; description: string }[]; defuzzifiers: string[]; }
interface Infer { output: number; fired_rules: { rule: string; firing: number }[]; }
interface Surface { x: number[]; y: number[]; z: number[][]; x_name: string; y_name: string; output: string; }
type Curves = Record<string, { x: number[]; terms: Record<string, number[]> }>;

const PALETTE = ["#7c6cff", "#2dd4bf", "#f59e0b", "#ef4444", "#38bdf8", "#a3e635"];
const SVGNS = "http://www.w3.org/2000/svg";

const $ = (id: string) => document.getElementById(id)!;

let desc: Describe;
let curves: Curves;
let surface: Surface | null = null;
const state: Record<string, number> = {};

// --- viridis-ish colour ramp ------------------------------------------------

function ramp(t: number): [number, number, number] {
  t = Math.min(1, Math.max(0, t));
  const r = 255 * Math.min(1, Math.max(0, 1.4 * t - 0.4));
  const g = 255 * Math.min(1, Math.max(0, 0.2 + 0.9 * t));
  const b = 255 * Math.min(1, Math.max(0, 0.85 - 0.95 * t));
  return [Math.round(r), Math.round(g), Math.round(b)];
}
const rgb = ([r, g, b]: [number, number, number]) => `rgb(${r},${g},${b})`;

// --- membership-function plots (SVG) ----------------------------------------

function mfPlot(v: VarInfo, marker: number | null): HTMLElement {
  const W = 300, H = 96, pad = 6;
  const data = curves[v.name];
  const xToPx = (x: number) => pad + ((x - v.low) / (v.high - v.low)) * (W - 2 * pad);
  const yToPx = (y: number) => H - pad - y * (H - 2 * pad);

  const wrap = document.createElement("div");
  wrap.className = "mf";
  const cap = document.createElement("div");
  cap.className = "cap";
  const title = document.createElement("strong");
  title.textContent = v.name;
  const legend = document.createElement("span");
  legend.className = "legend";
  cap.append(title, legend);

  const svg = document.createElementNS(SVGNS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  v.terms.forEach((term, i) => {
    const colour = PALETTE[i % PALETTE.length];
    const ys = data.terms[term];
    const pts = data.x.map((x, k) => `${xToPx(x).toFixed(1)},${yToPx(ys[k]).toFixed(1)}`).join(" ");
    const line = document.createElementNS(SVGNS, "polyline");
    line.setAttribute("points", pts);
    line.setAttribute("fill", "none");
    line.setAttribute("stroke", colour);
    line.setAttribute("stroke-width", "2");
    svg.appendChild(line);
    const tag = document.createElement("i");
    tag.style.color = colour;
    tag.textContent = term;
    legend.appendChild(tag);
  });

  if (marker !== null) {
    const mx = xToPx(marker);
    const line = document.createElementNS(SVGNS, "line");
    line.setAttribute("x1", String(mx)); line.setAttribute("x2", String(mx));
    line.setAttribute("y1", String(pad)); line.setAttribute("y2", String(H - pad));
    line.setAttribute("stroke", "#ffffff"); line.setAttribute("stroke-width", "1");
    line.setAttribute("stroke-dasharray", "3 3"); line.setAttribute("opacity", "0.6");
    svg.appendChild(line);
  }
  wrap.append(cap, svg);
  return wrap;
}

function renderMF() {
  const grid = $("mfgrid");
  grid.innerHTML = "";
  for (const v of desc.inputs) grid.appendChild(mfPlot(v, state[v.name]));
  grid.appendChild(mfPlot(desc.output, null));
}

// --- control-surface heatmap ------------------------------------------------

function renderSurface() {
  if (!surface) return;
  const canvas = $("surface") as HTMLCanvasElement;
  const ctx = canvas.getContext("2d")!;
  const { x, y, z } = surface;
  const nx = x.length, ny = y.length;
  let lo = Infinity, hi = -Infinity;
  for (const row of z) for (const v of row) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
  const span = hi - lo || 1;
  const cw = canvas.width / nx, ch = canvas.height / ny;
  for (let j = 0; j < ny; j++)
    for (let i = 0; i < nx; i++) {
      ctx.fillStyle = rgb(ramp((z[j][i] - lo) / span));
      ctx.fillRect(i * cw, canvas.height - (j + 1) * ch, cw + 1, ch + 1);
    }
  // operating point
  const xv = byName(surface.x_name), yv = byName(surface.y_name);
  const px = ((state[xv.name] - xv.low) / (xv.high - xv.low)) * canvas.width;
  const py = canvas.height - ((state[yv.name] - yv.low) / (yv.high - yv.low)) * canvas.height;
  ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.arc(px, py, 6, 0, Math.PI * 2); ctx.stroke();

  // colourbar + labels
  const scale = $("cscale");
  scale.style.background = `linear-gradient(to top, ${rgb(ramp(0))}, ${rgb(ramp(0.5))}, ${rgb(ramp(1))})`;
  $("cmin").textContent = lo.toFixed(1);
  $("cmax").textContent = hi.toFixed(1);
  $("axx").textContent = `x · ${surface.x_name}`;
  $("axy").textContent = `y · ${surface.y_name}`;
}

const byName = (n: string) => [desc.output, ...desc.inputs].find(v => v.name === n)!;

async function reloadSurface() {
  surface = await fuzzy<Surface>({
    action: "surface", n: 44,
    x: (($("surfx") as HTMLSelectElement).value),
    y: (($("surfy") as HTMLSelectElement).value),
  });
  renderSurface();
}

// --- inference --------------------------------------------------------------

async function refresh() {
  const res = await fuzzy<Infer>({ action: "infer", inputs: state });
  const o = desc.output;
  $("output").innerHTML = `${res.output.toFixed(2)}<small>${o.name} · ${o.low}–${o.high}</small>`;
  ($("gauge") as HTMLElement).style.width =
    `${(100 * (res.output - o.low)) / (o.high - o.low)}%`;
  $("rules").innerHTML = res.fired_rules.length
    ? res.fired_rules.map(r => `
        <div class="rule"><div class="txt">${r.rule}</div>
        <div class="bar"><span style="width:${(r.firing * 100).toFixed(0)}%"></span></div></div>`).join("")
    : "<em>no rules fired</em>";
  renderMF();
  renderSurface();
}

// --- build the system UI ----------------------------------------------------

function buildControls() {
  const root = $("controls");
  root.innerHTML = "";
  for (const v of desc.inputs) {
    state[v.name] = (v.low + v.high) / 2;
    const wrap = document.createElement("div");
    wrap.className = "control";
    wrap.innerHTML = `<div class="row"><span class="name">${v.name}</span>
      <span class="val" id="val-${v.name}">${state[v.name].toFixed(1)}</span></div>`;
    const slider = document.createElement("input");
    slider.type = "range";
    slider.min = String(v.low); slider.max = String(v.high);
    slider.step = String((v.high - v.low) / 250);
    slider.value = String(state[v.name]);
    slider.oninput = () => {
      state[v.name] = Number(slider.value);
      $(`val-${v.name}`).textContent = state[v.name].toFixed(1);
      refresh();
    };
    wrap.appendChild(slider);
    root.appendChild(wrap);
  }
}

function fillSurfaceSelectors() {
  const names = desc.inputs.map(v => v.name);
  for (const id of ["surfx", "surfy"]) {
    const sel = $(id) as HTMLSelectElement;
    sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join("");
  }
  (($("surfx") as HTMLSelectElement)).value = names[0];
  (($("surfy") as HTMLSelectElement)).value = names[1] ?? names[0];
  $("surfx").onchange = reloadSurface;
  $("surfy").onchange = reloadSurface;
}

async function loadSystem() {
  curves = await fuzzy<Curves>({ action: "curves" });
  buildControls();
  fillSurfaceSelectors();
  await reloadSurface();
  await refresh();
}

async function selectSystem() {
  const system = ($("system") as HTMLSelectElement).value;
  desc = await fuzzy<Describe>({ action: "select", system });
  ($("defuzz") as HTMLSelectElement).value = desc.defuzz;
  $("sysdesc").textContent = desc.name;
  await loadSystem();
}

// Apply an edited/new spec (the editor throws to surface build errors).
async function applySpec(spec: Spec) {
  desc = await fuzzy<Describe>({ action: "build", spec });
  ($("defuzz") as HTMLSelectElement).value = desc.defuzz;
  $("sysdesc").textContent = `${desc.name} (custom)`;
  await loadSystem();
}

async function openSpecEditor(blank: boolean) {
  const spec = blank ? null : await fuzzy<Spec>({ action: "get_spec" });
  openEditor(spec, applySpec);
}

async function main() {
  const info = await fuzzy<SystemsInfo>({ action: "systems" });
  ($("system") as HTMLSelectElement).innerHTML =
    info.systems.map(s => `<option value="${s.id}">${s.name}</option>`).join("");
  ($("defuzz") as HTMLSelectElement).innerHTML =
    info.defuzzifiers.map(d => `<option value="${d}">${d}</option>`).join("");

  desc = await fuzzy<Describe>({ action: "describe" });
  ($("defuzz") as HTMLSelectElement).value = desc.defuzz;
  $("sysdesc").textContent = desc.name;

  $("system").onchange = selectSystem;
  $("defuzz").onchange = async () => {
    const spec = await fuzzy<Spec>({ action: "get_spec" });
    spec.defuzz = ($("defuzz") as HTMLSelectElement).value;
    desc = await fuzzy<Describe>({ action: "build", spec });
    $("sysdesc").textContent = desc.name;
    await loadSystem();
  };
  $("newbtn").onclick = () => openSpecEditor(true);
  $("editbtn").onclick = () => openSpecEditor(false);

  await loadSystem();
}

main().catch(err => { $("output").textContent = `error: ${err}`; });
