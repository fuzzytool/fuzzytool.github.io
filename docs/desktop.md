# Desktop app (prototype)

<img src="../images/desktop_icon.png" alt="fuzzytool studio icon" width="96" align="right" />

**fuzzytool studio** is a small desktop app — a [Tauri](https://tauri.app)
shell (Rust + system webview) driving the `fuzzytool` library through a **Python
sidecar**. It turns the library into an interactive workbench: pick or **build**
a fuzzy system, move the inputs, and watch the output, the fired rules and the
control surface update live.

It lives in [`desktop/`](https://github.com/fuzzytool/fuzzytool.github.io/tree/main/desktop)
in the repository. It is a prototype: not published as a binary, but fully
runnable from source.

## What it does

- **Several built-in systems** — credit-risk premium, card-fraud alert, and an
  investment-risk advisor — selectable from a dropdown.
- **Live inference** — sliders for each input; a gauge shows the crisp output and
  a list shows which rules fired and how strongly (via
  [`integrations.agents.explain`](integrations.md#llm-agents)).
- **Membership-function plots** for every variable and a **control-surface
  heatmap** with X/Y variable pickers and a colour bar.
- **Configurable defuzzifier** (centroid, bisector, MOM/SOM/LOM), applied live.
- **A built-in editor** — create or edit variables, terms (`tri`/`trap`/`gauss`)
  and rules (clauses combined with AND/OR, optional NOT, a consequent and a
  weight), then **Apply** to compile and run the system instantly.

## How it works

```
 webview (sliders, plots, editor)
        │  invoke("fuzzy_call", json)
        ▼
   Rust (Tauri)  ──stdin/stdout, line-delimited JSON──▶  Python sidecar (fuzzytool)
```

Systems are described by a plain-JSON **spec** (variables + typed terms +
clause-based rules) that the sidecar compiles into a `Mamdani`. The built-in
demos are predefined specs, and the editor sends the same shape — so demos and
hand-built systems share one engine. The frozen-sidecar build path (PyInstaller)
lets a packaged app run with no Python installed.

## Run it

```bash
cd desktop
npm install

# Point the sidecar at a Python that has fuzzytool installed:
export FUZZY_SIDECAR_PYTHON=/path/to/venv/bin/python

npm run tauri dev
```

Requires Node, Rust with the [Tauri v2 prerequisites](https://tauri.app/start/prerequisites/),
and `fuzzytool` installed in the target Python. See the
[`desktop/README.md`](https://github.com/fuzzytool/fuzzytool.github.io/blob/main/desktop/README.md)
for the production build (frozen sidecar + `tauri build`).
