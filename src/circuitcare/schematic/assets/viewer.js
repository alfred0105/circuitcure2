const NS = "http://www.w3.org/2000/svg";
const instances = new WeakMap();

function make(name, attrs = {}) {
  const node = document.createElementNS(NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

export default function(component) {
  const { parentElement, data, setStateValue } = component;
  const root = parentElement.querySelector("[data-schematic-viewer]");
  const viewport = root.querySelector("[data-viewport]");
  const page = root.querySelector("[data-page]");
  const camera = root.querySelector("[data-camera]");
  const image = root.querySelector("[data-schematic-image]");
  const hitLayer = root.querySelector("[data-hit-regions]");
  const selectionLayer = root.querySelector("[data-selection-overlay]");
  const analysisLayer = root.querySelector("[data-analysis-overlay]");
  const zoomLabel = root.querySelector("[data-zoom-label]");
  const geometryLabel = root.querySelector("[data-geometry-label]");
  const empty = root.querySelector("[data-empty]");
  let state = instances.get(root);
  if (!state) {
    state = { scale: 1, x: 0, y: 0, pageKey: "", assetKey: "", objectUrl: "", selected: null, navigationId: null, drag: null, pointer: null, listeners: [], observer: null, eventCounter: 0 };
    instances.set(root, state);
  }
  state.eventCounter = Math.max(state.eventCounter, Number(data.event_sequence_seed) || 0);
  state.sendSelection = (region) => {
    state.eventCounter += 1;
    setStateValue("selection", {
      component_ref: region.component_ref,
      symbol_uuid: region.symbol_uuid,
      sheet_path: region.sheet_path,
      unit: region.unit,
      context_key: data.context_key,
      asset_key: data.asset_key,
      event_seq: state.eventCounter,
      event_id: `${Date.now()}-${state.eventCounter}`,
    });
  };
  state.data = data;
  const listen = (target, event, handler, options) => {
    target.addEventListener(event, handler, options);
    state.listeners.push(() => target.removeEventListener(event, handler, options));
  };
  const svgPoint = (clientX, clientY) => {
    const p = page.createSVGPoint(); p.x = clientX; p.y = clientY;
    return p.matrixTransform(page.getScreenCTM().inverse());
  };
  const renderCamera = () => {
    camera.setAttribute("transform", `translate(${state.x} ${state.y}) scale(${state.scale})`);
    zoomLabel.textContent = `${Math.round(state.scale * 100)}%`;
  };
  const fit = () => {
    const vb = data.view_box;
    const bounds = data.fit_view_box || vb;
    const baseScale = Math.min(viewport.clientWidth / vb[2], viewport.clientHeight / vb[3]);
    const fullPage = bounds[0] === vb[0] && bounds[1] === vb[1] &&
      bounds[2] === vb[2] && bounds[3] === vb[3];
    const contentScale = fullPage ? 1 : Math.min(
      viewport.clientWidth / (baseScale * bounds[2]),
      viewport.clientHeight / (baseScale * bounds[3]),
    ) * 0.9;
    state.scale = Math.min(8, Math.max(0.25, contentScale));
    state.x = vb[0] + vb[2] / 2 - (bounds[0] + bounds[2] / 2) * state.scale;
    state.y = vb[1] + vb[3] / 2 - (bounds[1] + bounds[3] / 2) * state.scale;
    state.userAdjusted = false;
    renderCamera();
  };
  const zoomAt = (factor, point) => {
    const scale = Math.min(8, Math.max(0.25, state.scale * factor));
    const ratio = scale / state.scale;
    state.x = point.x - (point.x - state.x) * ratio;
    state.y = point.y - (point.y - state.y) * ratio;
    state.scale = scale; state.userAdjusted = true; renderCamera();
  };
  const centerOn = (ref) => {
    const region = (data.regions || []).find((r) => r.component_ref === ref);
    if (!region) return;
    const points = region.polygon;
    const cx = points.reduce((sum, p) => sum + p[0], 0) / points.length;
    const cy = points.reduce((sum, p) => sum + p[1], 0) / points.length;
    const vb = data.view_box;
    state.x = vb[0] + vb[2] / 2 - cx * state.scale;
    state.y = vb[1] + vb[3] / 2 - cy * state.scale;
    renderCamera();
  };
  const buildRegions = () => {
    hitLayer.replaceChildren(); selectionLayer.replaceChildren();
    for (const [index, region] of (data.regions || []).entries()) {
      const shape = make("polygon", { points: region.polygon.map((p) => p.join(",")).join(" "),
        class: "cc-hit-region", "data-region-index": index, tabindex: "0", role: "button", "aria-label": `부품 ${region.component_ref} 선택` });
      hitLayer.appendChild(shape);
    }
    const selected = (data.regions || []).filter((r) => r.component_ref === data.selected_ref);
    for (const region of selected) {
      selectionLayer.appendChild(make("polygon", { points: region.polygon.map((p) => p.join(",")).join(" "), class: "cc-highlight" }));
    }
  };
  const buildOverlay = () => {
    analysisLayer.replaceChildren();
    const overlay = state.data.overlay;
    if (!overlay || overlay.status !== "READY") return;
    const regions = new Map((state.data.regions || []).map((region) => [region.component_ref, region]));
    const addText = (group, x, y, value, className) => {
      const label = make("text", { x, y, class: className, "text-anchor": "middle" });
      label.textContent = String(value);
      group.appendChild(label);
    };
    for (const marker of (overlay.markers || [])) {
      if (marker.kind === "MEASUREMENT_NODE_CANDIDATE") {
        const group = make("g", { class: "cc-overlay-measurement" });
        group.appendChild(make("circle", { cx: marker.x, cy: marker.y, r: 1.7 }));
        addText(group, marker.x + 2.4, marker.y - 1.2, marker.node, "cc-overlay-label");
        analysisLayer.appendChild(group);
        continue;
      }
      const region = regions.get(marker.component_ref);
      if (!region) continue;
      const points = region.polygon;
      const xs = points.map((point) => point[0]);
      const ys = points.map((point) => point[1]);
      const cx = (Math.min(...xs) + Math.max(...xs)) / 2;
      const cy = (Math.min(...ys) + Math.max(...ys)) / 2;
      const group = make("g", { class: "cc-overlay-marker" });
      if (marker.kind === "FAULT_SOURCE") {
        group.appendChild(make("polygon", { points: points.map((point) => point.join(",")).join(" "), class: "cc-overlay-fault" }));
        addText(group, cx, Math.min(...ys) - 1.2, "고장 가정", "cc-overlay-label");
      } else if (marker.kind === "DIRECT_CANDIDATE" || marker.kind === "INDIRECT_CANDIDATE") {
        const candidateClass = marker.kind === "DIRECT_CANDIDATE" ? "cc-overlay-direct" : "cc-overlay-indirect";
        group.appendChild(make("polygon", { points: points.map((point) => point.join(",")).join(" "), class: candidateClass }));
        addText(group, cx, Math.min(...ys) - 1.2, `${marker.kind === "DIRECT_CANDIDATE" ? "직접" : "간접"} · ${marker.hop_count}단계`, "cc-overlay-label");
      } else if (marker.kind === "RISK") {
        const levelClass = { EXCEEDED: "exceeded", NEAR_LIMIT: "near", WITHIN_LIMIT: "within", UNKNOWN: "unknown" }[marker.risk_level] || "unknown";
        const markerX = Math.max(...xs), markerY = Math.min(...ys);
        const symbol = { EXCEEDED: "▲", NEAR_LIMIT: "◆", WITHIN_LIMIT: "○", UNKNOWN: "?" }[marker.risk_level] || "?";
        const riskMark = make("text", { x: markerX, y: markerY, class: `cc-overlay-risk cc-overlay-risk-${levelClass}`, "text-anchor": "middle", "dominant-baseline": "central" });
        riskMark.textContent = symbol;
        group.appendChild(riskMark);
        const riskLabel = { EXCEEDED: "정격 초과", NEAR_LIMIT: "정격 근접", WITHIN_LIMIT: "정격 이내", UNKNOWN: "미평가" }[marker.risk_level] || "미평가";
        addText(group, cx, Math.min(...ys) - 1.6, riskLabel, "cc-overlay-label");
      } else if (marker.kind === "INSPECTION_PRIORITY") {
        group.appendChild(make("circle", { cx: Math.max(...xs), cy: Math.max(...ys), r: 2, class: "cc-overlay-inspection" }));
        addText(group, Math.max(...xs), Math.max(...ys) + 0.8, marker.priority_rank ?? "?", "cc-overlay-rank");
      }
      analysisLayer.appendChild(group);
    }
  };
  const setAsset = () => {
    const vb = data.view_box;
    if (!vb || !data.svg) {
      empty.hidden = false; page.hidden = true; return;
    }
    empty.hidden = true; page.hidden = false;
    page.setAttribute("viewBox", `${vb[0]} ${vb[1]} ${vb[2]} ${vb[3]}`);
    image.setAttribute("x", vb[0]); image.setAttribute("y", vb[1]);
    image.setAttribute("width", vb[2]); image.setAttribute("height", vb[3]);
    if (state.assetKey !== data.asset_key || !state.objectUrl) {
      if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = URL.createObjectURL(new Blob([data.svg], { type: "image/svg+xml" }));
      image.setAttribute("href", state.objectUrl);
      const newPage = state.assetKey !== data.asset_key;
      state.assetKey = data.asset_key;
      state.pageKey = data.asset_key;
      if (newPage) fit();
    }
    if (state.navigationId !== data.navigation_id) {
      state.navigationId = data.navigation_id;
      if (data.navigation_ref) centerOn(data.navigation_ref);
    }
  };
  const onWheel = (event) => {
    event.preventDefault();
    zoomAt(event.deltaY < 0 ? 1.12 : 1 / 1.12, svgPoint(event.clientX, event.clientY));
  };
  const onPointerDown = (event) => {
    if (event.button !== 0 || event.target.closest(".cc-hit-region")) return;
    const point = svgPoint(event.clientX, event.clientY);
    state.drag = { pointerId: event.pointerId, x: point.x, y: point.y, tx: state.x, ty: state.y };
    viewport.setPointerCapture(event.pointerId); viewport.dataset.dragging = "true";
  };
  const onPointerMove = (event) => {
    if (!state.drag || state.drag.pointerId !== event.pointerId) return;
    const point = svgPoint(event.clientX, event.clientY);
    state.x = state.drag.tx + point.x - state.drag.x;
    state.y = state.drag.ty + point.y - state.drag.y;
    state.userAdjusted = true;
    renderCamera();
  };
  const endDrag = (event) => {
    if (event?.pointerId !== undefined && state.drag && state.drag.pointerId !== event.pointerId) return;
    state.drag = null; viewport.dataset.dragging = "false";
  };
  if (state.listeners.length === 0) {
    listen(viewport, "wheel", onWheel, { passive: false });
    listen(viewport, "pointerdown", onPointerDown);
    listen(viewport, "pointermove", onPointerMove);
    listen(viewport, "pointerup", endDrag);
    listen(viewport, "pointercancel", endDrag);
    listen(viewport, "lostpointercapture", endDrag);
    listen(window, "blur", endDrag);
    listen(window, "pagehide", endDrag);
    listen(hitLayer, "click", (event) => {
      const shape = event.target.closest("[data-region-index]");
      if (!shape) return;
      event.stopPropagation();
      const region = state.data.regions?.[Number(shape.dataset.regionIndex)];
      if (region) state.sendSelection(region);
    });
    listen(hitLayer, "keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const shape = event.target.closest("[data-region-index]");
      if (!shape) return;
      event.preventDefault();
      const region = state.data.regions?.[Number(shape.dataset.regionIndex)];
      if (region) state.sendSelection(region);
    });
    for (const button of root.querySelectorAll("[data-action]")) listen(button, "click", () => {
      const action = button.dataset.action;
      if (action === "fit" || action === "reset") fit();
      else if (action === "zoom-in" || action === "zoom-out") {
        const rect = page.getBoundingClientRect();
        zoomAt(action === "zoom-in" ? 1.25 : 0.8, svgPoint(rect.left + rect.width / 2, rect.top + rect.height / 2));
      }
    });
    state.observer = new ResizeObserver(() => { if (!state.userAdjusted) fit(); });
    state.observer.observe(viewport);
  }
  if (state.pageKey !== data.asset_key) setAsset(); else setAsset();
  geometryLabel.textContent = data.geometry_status === "VERIFIED_GEOMETRY" ? "심벌 선택 · 검증됨" : `심벌 선택 · ${data.geometry_status}`;
  if (state.selected !== data.selected_ref || state.assetKey !== data.asset_key) {
    state.selected = data.selected_ref;
    buildRegions();
  } else {
    // Rebuilding keeps the Python → canvas highlight synchronized after a rerun.
    buildRegions();
  }
  buildOverlay();
  renderCamera();
  return () => {
    for (const remove of state.listeners.splice(0)) remove();
    state.observer?.disconnect();
    if (state.objectUrl) URL.revokeObjectURL(state.objectUrl);
    state.objectUrl = "";
  };
}
