"""A 2D click-to-pick slider as an anywidget.

Click anywhere on the plane to set (x_value, y_value); both traits are
synced to Python. Designed for the singular-gradient lead cell of
notebooks/walkthrough.py: the user picks a probe point on the (u_1, u_2)
plane and the figure cell recomputes its red overlay curve in closed
form on every click.

Visual aids:
- Cross-hair guides at the current probe.
- 4 grey markers at the discrete data atoms (corners of [-1, 1]^2),
  matching the centres used by data/singular_gradient.npz.

Reference: marimo gallery, "A 2D slider control for selecting x/y
coordinates interactively" -- adapted as a self-contained anywidget so
the notebook does not depend on a gallery import path.
"""

import anywidget
import traitlets


_ESM = r"""
function render({ model, el }) {
    const W = 220, H = 220;
    const ns = "http://www.w3.org/2000/svg";
    const svg = document.createElementNS(ns, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    svg.style.border = "1px solid #888";
    svg.style.background = "#fafafa";
    svg.style.cursor = "crosshair";

    const cross_v = document.createElementNS(ns, "line");
    cross_v.setAttribute("y1", 0); cross_v.setAttribute("y2", H);
    cross_v.setAttribute("stroke", "#bbb");
    cross_v.setAttribute("stroke-dasharray", "2,2");
    svg.appendChild(cross_v);

    const cross_h = document.createElementNS(ns, "line");
    cross_h.setAttribute("x1", 0); cross_h.setAttribute("x2", W);
    cross_h.setAttribute("stroke", "#bbb");
    cross_h.setAttribute("stroke-dasharray", "2,2");
    svg.appendChild(cross_h);

    const xRange = model.get("x_range");
    const yRange = model.get("y_range");

    function toPx(x, y) {
        const px = ((x - xRange[0]) / (xRange[1] - xRange[0])) * W;
        const py = H - ((y - yRange[0]) / (yRange[1] - yRange[0])) * H;
        return [px, py];
    }
    function toData(px, py) {
        const x = xRange[0] + (px / W) * (xRange[1] - xRange[0]);
        const y = yRange[0] + (1 - py / H) * (yRange[1] - yRange[0]);
        return [x, y];
    }

    // 4 corner markers for the discrete data atoms.
    for (const [dx, dy] of [[1, 1], [1, -1], [-1, 1], [-1, -1]]) {
        const [cx, cy] = toPx(dx, dy);
        const m = document.createElementNS(ns, "circle");
        m.setAttribute("cx", cx); m.setAttribute("cy", cy);
        m.setAttribute("r", 4); m.setAttribute("fill", "#888");
        svg.appendChild(m);
    }

    const dot = document.createElementNS(ns, "circle");
    dot.setAttribute("r", 6); dot.setAttribute("fill", "red");
    dot.setAttribute("stroke", "white"); dot.setAttribute("stroke-width", 1.5);
    svg.appendChild(dot);

    const label = document.createElement("div");
    label.style.fontFamily = "monospace";
    label.style.fontSize = "11px";
    label.style.marginTop = "4px";

    function refresh() {
        const x = model.get("x_value");
        const y = model.get("y_value");
        const [px, py] = toPx(x, y);
        dot.setAttribute("cx", px); dot.setAttribute("cy", py);
        cross_v.setAttribute("x1", px); cross_v.setAttribute("x2", px);
        cross_h.setAttribute("y1", py); cross_h.setAttribute("y2", py);
        label.textContent = `probe: (u1 = ${x.toFixed(2)}, u2 = ${y.toFixed(2)})`;
    }
    refresh();
    model.on("change:x_value", refresh);
    model.on("change:y_value", refresh);

    function pick(event) {
        const rect = svg.getBoundingClientRect();
        const px = (event.clientX - rect.left) * (W / rect.width);
        const py = (event.clientY - rect.top) * (H / rect.height);
        const [x, y] = toData(px, py);
        model.set("x_value", x);
        model.set("y_value", y);
        model.save_changes();
    }
    svg.addEventListener("click", pick);

    el.appendChild(svg);
    el.appendChild(label);
}
export default { render };
"""


class TwoDSliderWidget(anywidget.AnyWidget):
    _esm = _ESM

    x_value = traitlets.Float(1.5).tag(sync=True)
    y_value = traitlets.Float(0.0).tag(sync=True)
    x_range = traitlets.List(traitlets.Float(), default_value=[-3.0, 3.0]).tag(sync=True)
    y_range = traitlets.List(traitlets.Float(), default_value=[-3.0, 3.0]).tag(sync=True)

    def __init__(self, x_range=None, y_range=None, x_value=1.5, y_value=0.0, **kwargs):
        traits: dict = {"x_value": float(x_value), "y_value": float(y_value)}
        if x_range is not None:
            traits["x_range"] = [float(v) for v in x_range]
        if y_range is not None:
            traits["y_range"] = [float(v) for v in y_range]
        super().__init__(**traits, **kwargs)
