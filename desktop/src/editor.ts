// A small modal editor for building/editing a fuzzy-system spec (variables,
// terms and rules). It mutates a working copy and calls `onApply` with the spec;
// the sidecar compiles it. Re-renders on structural changes; plain text/number
// edits update the model in place to preserve focus.

export type MFType = "tri" | "trap" | "gauss";
export interface Term { name: string; type: MFType; params: number[]; }
export interface Var { name: string; low: number; high: number; role: "input" | "output"; terms: Term[]; }
export interface Clause { var: string; term: string; negate?: boolean; }
export interface Rule { op: "and" | "or"; clauses: Clause[]; consequent: { var: string; term: string }; weight?: number; }
export interface Spec { name: string; defuzz: string; variables: Var[]; rules: Rule[]; }

const MF_PARAMS: Record<MFType, number> = { tri: 3, trap: 4, gauss: 2 };
const DEFUZZ = ["centroid", "bisector", "mom", "som", "lom"];

const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));
const esc = (s: string) => s.replace(/"/g, "&quot;");

function blankSpec(): Spec {
  return {
    name: "New system", defuzz: "centroid",
    variables: [
      { name: "x", low: 0, high: 10, role: "input",
        terms: [{ name: "low", type: "tri", params: [0, 0, 5] }, { name: "high", type: "tri", params: [5, 10, 10] }] },
      { name: "y", low: 0, high: 1, role: "output",
        terms: [{ name: "small", type: "tri", params: [0, 0, 0.5] }, { name: "big", type: "tri", params: [0.5, 1, 1] }] },
    ],
    rules: [],
  };
}

export function openEditor(initial: Spec | null, onApply: (spec: Spec) => Promise<void>) {
  const spec: Spec = initial ? clone(initial) : blankSpec();

  const overlay = document.createElement("div");
  overlay.className = "modal-overlay";
  const modal = document.createElement("div");
  modal.className = "modal";
  overlay.appendChild(modal);
  document.body.appendChild(overlay);

  const close = () => overlay.remove();

  function termRow(vi: number, ti: number, t: Term): string {
    const params = Array.from({ length: MF_PARAMS[t.type] }, (_, pi) =>
      `<input class="num" type="number" step="any" data-f="tparam" data-vi="${vi}" data-ti="${ti}" data-pi="${pi}" value="${t.params[pi] ?? 0}">`).join("");
    const opt = (k: MFType) => `<option value="${k}" ${t.type === k ? "selected" : ""}>${k}</option>`;
    return `<div class="termrow">
      <input class="t-name" data-f="tname" data-vi="${vi}" data-ti="${ti}" value="${esc(t.name)}">
      <select data-f="ttype" data-vi="${vi}" data-ti="${ti}">${opt("tri")}${opt("trap")}${opt("gauss")}</select>
      <div class="params">${params}</div>
      <button class="icon" data-act="del-term" data-vi="${vi}" data-ti="${ti}">✕</button>
    </div>`;
  }

  function varCard(vi: number, v: Var): string {
    const role = (r: string) => `<option value="${r}" ${v.role === r ? "selected" : ""}>${r}</option>`;
    return `<div class="ed-card">
      <div class="ed-row">
        <input class="v-name" data-f="vname" data-vi="${vi}" value="${esc(v.name)}">
        <select data-f="vrole" data-vi="${vi}">${role("input")}${role("output")}</select>
        <label>min <input class="num" type="number" step="any" data-f="vlow" data-vi="${vi}" value="${v.low}"></label>
        <label>max <input class="num" type="number" step="any" data-f="vhigh" data-vi="${vi}" value="${v.high}"></label>
        <button class="icon" data-act="del-var" data-vi="${vi}">✕</button>
      </div>
      ${v.terms.map((t, ti) => termRow(vi, ti, t)).join("")}
      <button class="add" data-act="add-term" data-vi="${vi}">+ term</button>
    </div>`;
  }

  function clauseRow(ri: number, ci: number, c: Clause): string {
    const vars = spec.variables.filter(v => v.role === "input");
    const terms = spec.variables.find(v => v.name === c.var)?.terms ?? [];
    const vOpt = vars.map(v => `<option ${v.name === c.var ? "selected" : ""}>${v.name}</option>`).join("");
    const tOpt = terms.map(t => `<option ${t.name === c.term ? "selected" : ""}>${t.name}</option>`).join("");
    return `<div class="clause">
      <label class="chk"><input type="checkbox" data-f="cneg" data-ri="${ri}" data-ci="${ci}" ${c.negate ? "checked" : ""}> not</label>
      <select data-f="cvar" data-ri="${ri}" data-ci="${ci}">${vOpt}</select>
      <span class="is">is</span>
      <select data-f="cterm" data-ri="${ri}" data-ci="${ci}">${tOpt}</select>
      <button class="icon" data-act="del-clause" data-ri="${ri}" data-ci="${ci}">✕</button>
    </div>`;
  }

  function ruleCard(ri: number, r: Rule): string {
    const outs = spec.variables.filter(v => v.role === "output");
    const cv = spec.variables.find(v => v.name === r.consequent.var) ?? outs[0];
    const op = (o: string) => `<option value="${o}" ${r.op === o ? "selected" : ""}>${o.toUpperCase()}</option>`;
    const vOpt = outs.map(v => `<option ${v.name === r.consequent.var ? "selected" : ""}>${v.name}</option>`).join("");
    const tOpt = (cv?.terms ?? []).map(t => `<option ${t.name === r.consequent.term ? "selected" : ""}>${t.name}</option>`).join("");
    return `<div class="ed-card">
      <div class="ed-row"><strong>IF</strong>
        <select data-f="rop" data-ri="${ri}">${op("and")}${op("or")}</select>
        <span class="muted">combine clauses</span>
        <button class="icon" style="margin-left:auto" data-act="del-rule" data-ri="${ri}">✕</button>
      </div>
      ${r.clauses.map((c, ci) => clauseRow(ri, ci, c)).join("")}
      <button class="add" data-act="add-clause" data-ri="${ri}">+ clause</button>
      <div class="ed-row" style="margin-top:.5rem"><strong>THEN</strong>
        <select data-f="rcvar" data-ri="${ri}">${vOpt}</select><span class="is">is</span>
        <select data-f="rcterm" data-ri="${ri}">${tOpt}</select>
        <label>weight <input class="num" type="number" step="0.05" min="0" max="1" data-f="rweight" data-ri="${ri}" value="${r.weight ?? 1}"></label>
      </div>
    </div>`;
  }

  function render() {
    const dz = DEFUZZ.map(d => `<option value="${d}" ${spec.defuzz === d ? "selected" : ""}>${d}</option>`).join("");
    modal.innerHTML = `
      <div class="modal-head">
        <input class="title" data-f="name" value="${esc(spec.name)}">
        <select data-f="defuzz">${dz}</select>
        <button class="icon" data-act="cancel">✕</button>
      </div>
      <div class="modal-body">
        <h3>Variables</h3>
        ${spec.variables.map((v, vi) => varCard(vi, v)).join("")}
        <button class="add" data-act="add-var">+ variable</button>
        <h3>Rules</h3>
        ${spec.rules.map((r, ri) => ruleCard(ri, r)).join("")}
        <button class="add" data-act="add-rule">+ rule</button>
      </div>
      <div class="modal-foot">
        <span class="err" id="ed-err"></span>
        <button class="btn ghost" data-act="cancel">Cancel</button>
        <button class="btn primary" data-act="apply">Apply</button>
      </div>`;
  }

  // --- delegated handlers ---
  const num = (el: HTMLInputElement) => Number(el.value);
  const ds = (el: HTMLElement) => el.dataset;

  modal.addEventListener("input", e => {
    const el = e.target as HTMLInputElement;
    const f = el.dataset.f; if (!f) return;
    const vi = Number(el.dataset.vi), ti = Number(el.dataset.ti), pi = Number(el.dataset.pi), ri = Number(el.dataset.ri);
    if (f === "name") spec.name = el.value;
    else if (f === "vname") spec.variables[vi].name = el.value;
    else if (f === "vlow") spec.variables[vi].low = num(el);
    else if (f === "vhigh") spec.variables[vi].high = num(el);
    else if (f === "tname") spec.variables[vi].terms[ti].name = el.value;
    else if (f === "tparam") spec.variables[vi].terms[ti].params[pi] = num(el);
    else if (f === "rweight") spec.rules[ri].weight = num(el);
  });

  modal.addEventListener("change", e => {
    const el = e.target as HTMLInputElement | HTMLSelectElement;
    const f = (el as HTMLElement).dataset.f; if (!f) return;
    const d = ds(el), vi = Number(d.vi), ti = Number(d.ti), ri = Number(d.ri), ci = Number(d.ci);
    const val = (el as HTMLInputElement).value;
    if (f === "defuzz") spec.defuzz = val;
    else if (f === "vrole") { spec.variables[vi].role = val as Var["role"]; render(); }
    else if (f === "ttype") {
      const t = spec.variables[vi].terms[ti];
      t.type = val as MFType;
      const n = MF_PARAMS[t.type];
      t.params = Array.from({ length: n }, (_, i) => t.params[i] ?? 0);
      render();
    }
    else if (f === "rop") spec.rules[ri].op = val as Rule["op"];
    else if (f === "cvar") { spec.rules[ri].clauses[ci].var = val; spec.rules[ri].clauses[ci].term = ""; render(); }
    else if (f === "cterm") spec.rules[ri].clauses[ci].term = val;
    else if (f === "cneg") spec.rules[ri].clauses[ci].negate = (el as HTMLInputElement).checked;
    else if (f === "rcvar") { spec.rules[ri].consequent.var = val; spec.rules[ri].consequent.term = ""; render(); }
    else if (f === "rcterm") spec.rules[ri].consequent.term = val;
  });

  modal.addEventListener("click", async e => {
    const btn = (e.target as HTMLElement).closest("[data-act]") as HTMLElement | null;
    if (!btn) return;
    const act = btn.dataset.act!;
    const vi = Number(btn.dataset.vi), ri = Number(btn.dataset.ri), ci = Number(btn.dataset.ci), ti = Number(btn.dataset.ti);
    const inputs = spec.variables.filter(v => v.role === "input");
    const outputs = spec.variables.filter(v => v.role === "output");
    if (act === "cancel") return close();
    if (act === "add-var") spec.variables.push({ name: `var${spec.variables.length + 1}`, low: 0, high: 1, role: "input", terms: [{ name: "low", type: "tri", params: [0, 0, 0.5] }] });
    else if (act === "del-var") spec.variables.splice(vi, 1);
    else if (act === "add-term") spec.variables[vi].terms.push({ name: `t${spec.variables[vi].terms.length + 1}`, type: "tri", params: [0, 0, 1] });
    else if (act === "del-term") spec.variables[vi].terms.splice(ti, 1);
    else if (act === "add-rule") spec.rules.push({ op: "and", clauses: inputs[0] ? [{ var: inputs[0].name, term: inputs[0].terms[0]?.name ?? "" }] : [], consequent: { var: outputs[0]?.name ?? "", term: outputs[0]?.terms[0]?.name ?? "" }, weight: 1 });
    else if (act === "del-rule") spec.rules.splice(ri, 1);
    else if (act === "add-clause") spec.rules[ri].clauses.push({ var: inputs[0]?.name ?? "", term: inputs[0]?.terms[0]?.name ?? "" });
    else if (act === "del-clause") spec.rules[ri].clauses.splice(ci, 1);
    else if (act === "apply") {
      const err = modal.querySelector("#ed-err") as HTMLElement;
      err.textContent = "";
      try { await onApply(clone(spec)); close(); }
      catch (ex) { err.textContent = String(ex instanceof Error ? ex.message : ex); }
      return;
    } else return;
    render();
  });

  render();
}
