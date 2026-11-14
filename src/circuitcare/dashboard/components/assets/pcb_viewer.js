const SVG_NS = "http://www.w3.org/2000/svg";
const DRAG_THRESHOLD_PX = 5;
const instances = new WeakMap();

function svgElement(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

export default function(component) {
  const { parentElement, data, setStateValue, setTriggerValue } = component;
  const root = parentElement.querySelector("[data-pcb-viewer]");
  const viewport = root.querySelector("[data-viewport]");
  const canvas = root.querySelector("[data-canvas]");
  const camera = root.querySelector("[data-camera]");
  const image = root.querySelector("[data-board-image]");
  const layer = root.querySelector("[data-component-layer]");
  const tooltip = root.querySelector("[data-tooltip]");
  const empty = root.querySelector("[data-empty]");
  const zoomLabel = root.querySelector("[data-zoom-label]");
  let state = instances.get(root);
  if (!state) {
    state = { scale:1, x:0, y:0, assetKey:"", hover:null, drag:null, drawMode:false, boxDraw:null,
      selectedComponentIds:[],
      selectedComponentId:undefined, pendingSelection:null,
      listeners:[], observer:null, sequence:0, userAdjusted:false, mountedAssetKey:"",
      instanceId:`pcb-${Date.now()}-${Math.random().toString(36).slice(2,8)}`, renderCount:0, hrefChangeCount:0 };
    instances.set(root, state);
    state.mounted=true;
  }
  state.renderCount += 1;
  state.data = data;
  const loading = Boolean(data.analysis_loading);
  root.dataset.loading = loading ? "true" : "false";
  const loadingPanel = root.querySelector("[data-loading]");
  if (loadingPanel) loadingPanel.hidden = !loading;
  if (loading) { tooltip.hidden = true; state.hover = null; }
  const telemetry = (event, details = {}) => {
    if (!data.debug_telemetry) return;
    const digest = (value) => {
      let hash=2166136261;
      for (const char of String(value || "")) hash=Math.imul(hash ^ char.charCodeAt(0),16777619);
      return (hash>>>0).toString(16).padStart(8,"0");
    };
    console.debug("[pcb-viewer-telemetry]", {event, timestamp:new Date().toISOString(),
      instance_id:state.instanceId, viewer_key:root.dataset.viewerKey || data.viewer_key || null,
      project_id:data.project_id, photo_id:data.photo_id, asset_key_hash:digest(state.assetKey || data.asset_key), frame_id:data.frame_id,
      image_fingerprint:String(data.asset_key || "").slice(0,16), render_count:state.renderCount,
      href_change_count:state.hrefChangeCount, ...details});
  };
  if (state.renderCount === 1) telemetry("VIEWER_MOUNT", {reason:"component identity created"});
  else telemetry("VIEWER_RENDER", {reason:"Streamlit component data update"});
  state.setStateValue = setStateValue;
  state.setTriggerValue = setTriggerValue;
  // Hold the optimistic highlight until Python acknowledges this exact click.
  // A rerun carrying the previous server selection must not make it flash off.
  if (state.pendingSelection && data.selection_ack_event_id === state.pendingSelection.eventId) {
    telemetry("SELECTION_ACK", {event_id:state.pendingSelection.eventId, component_id:state.pendingSelection.componentId});
    state.pendingSelection = null;
    state.selectedComponentId = data.selected_component_id;
  } else if (state.pendingSelection && data.selection_rejected_event_id === state.pendingSelection.eventId) {
    state.pendingSelection = null;
    state.selectedComponentId = data.selected_component_id;
  } else if (!state.pendingSelection) {
    state.selectedComponentId = data.selected_component_id;
  }
  state.selectedComponentIds = Array.isArray(data.selected_component_ids) ? data.selected_component_ids : [];
  state.sequence = Math.max(state.sequence, Number(data.event_sequence_seed) || 0);
  const listen = (target, event, callback, options) => {
    target.addEventListener(event, callback, options);
    state.listeners.push(() => target.removeEventListener(event, callback, options));
  };
  const point = (clientX, clientY) => {
    const p = canvas.createSVGPoint(); p.x = clientX; p.y = clientY;
    return p.matrixTransform(canvas.getScreenCTM().inverse());
  };
  // Hover and click must use this exact SVG→image transform and hit policy.
  const hitComponents = (event) => {
    if (state.data.analysis_loading) return [];
    const p = point(event.clientX, event.clientY);
    const x = (p.x - state.x) / state.scale;
    const y = (p.y - state.y) / state.scale;
    return (state.data.components || []).filter((item) => {
      const [x1, y1, x2, y2] = item.bbox_xyxy;
      return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    }).sort((a, b) => {
      // Never let the previous selection bias a new physical click. Nested boxes
      // resolve to the smallest hit; the overlap picker can still choose a parent.
      const aArea = (a.bbox_xyxy[2] - a.bbox_xyxy[0]) * (a.bbox_xyxy[3] - a.bbox_xyxy[1]);
      const bArea = (b.bbox_xyxy[2] - b.bbox_xyxy[0]) * (b.bbox_xyxy[3] - b.bbox_xyxy[1]);
      return aArea - bArea || a.component_id.localeCompare(b.component_id);
    });
  };
  const hitComponent = (event) => hitComponents(event)[0] || null;
  const drawCamera = () => {
    camera.setAttribute("transform", `translate(${state.x} ${state.y}) scale(${state.scale})`);
    zoomLabel.textContent = `${Math.round(state.scale * 100)}%`;
  };
  const fit = () => {
    const [width, height] = data.image_size_px;
    if (!width || !height || !viewport.clientWidth || !viewport.clientHeight) return;
    // SVG preserveAspectRatio performs centered fitting once for both image and vector layer.
    state.scale = 1; state.x = 0; state.y = 0; state.userAdjusted = false; drawCamera();
  };
  const zoomAt = (factor, p) => {
    const scale = Math.max(0.25, Math.min(8, state.scale * factor));
    const ratio = scale / state.scale;
    state.x = p.x - (p.x - state.x) * ratio;
    state.y = p.y - (p.y - state.y) * ratio;
    state.scale = scale; state.userAdjusted = true; drawCamera();
  };
  const applySelectedHighlight = (componentId) => {
    const previous = state.selectedComponentId || null;
    state.selectedComponentId = componentId || null;
    if (previous && previous !== state.selectedComponentId)
      state.regionNodes?.get(previous)?.shape.classList.remove("cc-selected");
    if (state.selectedComponentId)
      state.regionNodes?.get(state.selectedComponentId)?.shape.classList.add("cc-selected");
    const statusLabel = root.querySelector("[data-status-label]");
    const selected = (state.data.components || []).find((item) => item.component_id === state.selectedComponentId);
    if (statusLabel) statusLabel.textContent = (state.selectedComponentIds || []).length > 1
      ? `다중 선택 ${(state.selectedComponentIds || []).length}개 · ⌘/Ctrl+클릭으로 추가·해제`
      : selected ? `선택 ${selected.display_label || selected.component_type} · ${selected.component_id}`
        : `박스 ${(state.data.components || []).length}개 · 클릭하여 선택`;
  };
  const sendSelection = (componentId, action = null, bbox = null, overlapIds = null) => {
    if (state.data.analysis_loading) return;
    state.sequence += 1;
    const eventId = `${Date.now()}-${state.sequence}`;
    const payload = {
      project_id:state.data.project_id, board_id:state.data.board_id, board_side:state.data.board_side,
      component_id:componentId, canonical_frame_id:state.data.frame_id,
      selection_action:action || (componentId ? "select" : "clear"), bbox_xyxy:bbox, event_seq:state.sequence,
      event_epoch:state.data.event_epoch,
      overlapping_component_ids:overlapIds || (componentId ? [componentId] : []),
      event_id:eventId,
    };
    telemetry("SELECTION_SENT", {event_id:payload.event_id, event_seq:payload.event_seq,
      component_id:componentId, overlapping_count:payload.overlapping_component_ids.length});
    console.debug("[pcb-selection] event_send", {event_id:payload.event_id, seq:payload.event_seq,
      action:payload.selection_action, component_id:componentId,
      overlapping_component_ids:payload.overlapping_component_ids});
    // Keep the latest pointer event in component state. A unique event_id makes
    // repeated clicks distinct state updates, while the state callback is
    // reliably scoped to the owning Streamlit fragment. Trigger aggregators
    // can be delivered after a fragment has already rerun, leaving the browser
    // highlight ahead of the Python inspector and manual-box store.
    state.setStateValue("selection_event", payload);
    if (action !== "draw") state.pendingSelection = {eventId, componentId};
  };
  const showTooltip = (item, event) => {
    const typeLabels = { capacitor:"커패시터", resistor:"저항", inductor:"인덕터", diode:"다이오드",
      transistor:"트랜지스터", integrated_circuit:"IC", ic:"IC", connector:"커넥터",
      unknown_component:"부품 후보" };
    const rows = [typeLabels[item.component_type?.toLowerCase()] || item.component_type];
    if (item.display_label) rows.push(`Reference ${item.display_label}`);
    rows.push(`식별 ${item.display_status || "미확인"}`);
    if (item.fault_role_status === "INFERRED") rows.push("기능 역할 추정 · 미확정");
    else if (item.fault_role_status === "CONFIRMED") rows.push("기능 역할 확인됨");
    if (item.ocr_marking) rows.push(`표기 ${item.ocr_marking}`);
    rows.push("클릭하면 오른쪽 패널에서 박스를 수정하거나 제외할 수 있습니다.");
    const faultLabels = { SUSPECTED_FAULT:"고장 의심 후보 · 검사 필요", RELATED_COMPONENT:"관련 점검 후보",
      CONFIRMED_FAULT:"검사 근거 확인", FAULT_SUPPORTED:"검증 검사 · 고장 가설 지지",
      FAULT_REFUTED:"검증 검사 · 고장 가설 반박", INCONCLUSIVE:"검사 판정 불가",
      CONFLICTED:"측정 결과 충돌 · 판단 보류" };
    if (faultLabels[item.fault_candidate_status]) {
      rows.push(faultLabels[item.fault_candidate_status]);
    } else {
      const diagnosisLabels = {
        DIAGNOSIS_NOT_RUN:"고장 진단 미실행",
        DIAGNOSIS_UNAVAILABLE:"진단 불가 · 분석 조건을 확인하세요",
        DIAGNOSIS_RESULT_NONE:"진단 결과 없음",
        UNDETERMINED:"고장 여부 판단 불가",
        MODEL_MISMATCH:"진단 모델과 측정값 불일치\n고장 여부 판단 불가",
      };
      rows.push(diagnosisLabels[item.diagnosis_display_status] || "고장 진단 미실행");
    }
    for (const history of item.fault_history || []) {
      const historyLabel = faultLabels[history.visual_status];
      if (historyLabel) rows.push(`이전 검사 · ${historyLabel}`);
    }
    tooltip.textContent = rows.join("\n"); tooltip.hidden = false;
    const rect = viewport.getBoundingClientRect();
    tooltip.style.left = `${Math.min(Math.max(8, event.clientX - rect.left + 12), rect.width - 220)}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - rect.top + 12)}px`;
  };
  const renderRegions = () => {
    const components = data.components || [];
    const selectedId = state.selectedComponentId !== undefined
      ? state.selectedComponentId : data.selected_component_id;
    const bulkIds = new Set(state.selectedComponentIds || []);
    const overlayChanged = state.renderedOverlayVersion !== data.overlay_version;
    const bulkSignature = [...bulkIds].sort().join("|");
    const selectionChanged = state.renderedSelectedId !== selectedId
      || state.renderedBulkSignature !== bulkSignature;
    if (!overlayChanged && !selectionChanged) return;
    if (!overlayChanged) {
      if (state.renderedSelectedId)
        state.regionNodes?.get(state.renderedSelectedId)?.shape.classList.remove("cc-selected");
      for (const id of state.renderedBulkIds || [])
        state.regionNodes?.get(id)?.shape.classList.remove("cc-bulk-selected");
      if (selectedId)
        state.regionNodes?.get(selectedId)?.shape.classList.add("cc-selected");
      for (const id of bulkIds) state.regionNodes?.get(id)?.shape.classList.add("cc-bulk-selected");
      state.renderedBulkIds = [...bulkIds];
      state.renderedBulkSignature = bulkSignature;
      state.renderedSelectedId = selectedId;
      const statusLabel = root.querySelector("[data-status-label]");
      const selected = components.find((item) => item.component_id === selectedId);
      if (statusLabel) statusLabel.textContent = bulkIds.size > 1
        ? `다중 선택 ${bulkIds.size}개 · ⌘/Ctrl+클릭으로 추가·해제`
        : selected ? `선택 ${selected.display_label || selected.component_type} · ${selected.component_id}`
          : `박스 ${components.length}개 · 클릭하여 선택`;
      return;
    }
    if (!state.regionNodes) state.regionNodes = new Map();
    const desired = new Set(components.map((item) => item.component_id));
    for (const [id, entry] of state.regionNodes) {
      if (!desired.has(id)) {
        for (const node of entry.nodes) node.remove();
        state.regionNodes.delete(id);
      }
    }
    for (const item of components) {
      const id = item.component_id;
      const [x1,y1,x2,y2] = item.bbox_xyxy;
      const visualKey = JSON.stringify([item.bbox_xyxy, item.display_label, item.component_type,
        item.overlay_status, item.overlay_source, item.fault_candidate_status, item.functional_block_roles]);
      let entry = state.regionNodes.get(id);
      if (!entry) {
        const shape = svgElement("rect", { "data-component-id":id, tabindex:"0", role:"button" });
        const label = svgElement("text", { class:"cc-pcb-label" });
        const conflict = svgElement("text", { class:"cc-pcb-conflict-icon" });
        layer.append(shape, label, conflict);
        entry = { nodes:[shape,label,conflict], shape, label, conflict, visualKey:null };
        state.regionNodes.set(id, entry);
      }
      if (entry.visualKey !== visualKey) {
        const shape = entry.shape;
        shape.setAttribute("x", String(x1)); shape.setAttribute("y", String(y1));
        shape.setAttribute("width", String(x2-x1)); shape.setAttribute("height", String(y2-y1));
        shape.setAttribute("aria-label", `${item.display_label || item.component_type} 부품 선택`);
        const classes = ["cc-pcb-component"];
        if (item.overlay_status === "EXCLUDED") classes.push("cc-excluded");
        if (item.overlay_source === "qwen_suggestion") classes.push("cc-ai-suggested");
        if (item.overlay_source === "yolo_second_pass") classes.push("cc-second-pass");
        if (item.overlay_source === "manual") classes.push("cc-manual-box");
        const blockRoleClasses = { power:"cc-block-power", control:"cc-block-control",
          control_timing:"cc-block-control", control_input:"cc-block-input", input:"cc-block-input",
          output:"cc-block-output", driver:"cc-block-driver", protection:"cc-block-protection",
          thermal:"cc-block-protection", buzzer:"cc-block-output", relay:"cc-block-output" };
        for (const role of (item.functional_block_roles || [])) {
          if (blockRoleClasses[role]) classes.push(blockRoleClasses[role]);
        }
        const faultClass = { SUSPECTED_FAULT:"cc-fault-suspected", RELATED_COMPONENT:"cc-fault-related",
          CONFIRMED_FAULT:"cc-fault-confirmed", FAULT_SUPPORTED:"cc-fault-supported",
          FAULT_REFUTED:"cc-fault-refuted", INCONCLUSIVE:"cc-fault-inconclusive" }[item.fault_candidate_status];
        if (faultClass) classes.push(faultClass);
        if (item.fault_candidate_status === "CONFLICTED") classes.push("cc-fault-conflicted");
        shape.setAttribute("class", classes.join(" "));
        entry.label.setAttribute("x", String((x1+x2)/2));
        entry.label.setAttribute("y", String(Math.max(12,y1-4)));
        entry.label.textContent = item.display_label || "";
        entry.conflict.setAttribute("x", String(x1+4));
        entry.conflict.setAttribute("y", String(y1+14));
        entry.conflict.textContent = item.fault_candidate_status === "CONFLICTED" ? "⚠" : "";
        entry.visualKey = visualKey;
      }
      entry.shape.classList.toggle("cc-selected", id === selectedId);
      entry.shape.classList.toggle("cc-bulk-selected", bulkIds.has(id));
    }
    state.renderedOverlayVersion = data.overlay_version;
    state.renderedSelectedId = selectedId;
    state.renderedBulkIds = [...bulkIds];
    state.renderedBulkSignature = bulkSignature;
    const statusLabel = root.querySelector("[data-status-label]");
    if (statusLabel) {
      const selected = components.find((item) => item.component_id === selectedId);
      statusLabel.textContent = bulkIds.size > 1
        ? `다중 선택 ${bulkIds.size}개 · ⌘/Ctrl+클릭으로 추가·해제`
        : selected ? `선택 ${selected.display_label || selected.component_type} · ${selected.component_id}`
          : `박스 ${components.length}개 · 클릭하여 선택`;
    }
  };
  const setImage = () => {
    const [width,height] = data.image_size_px || [];
    if (!data.image_png_base64 || !width || !height) { empty.hidden=false; return; }
    empty.hidden=true;
    const key = `${data.asset_key}|${data.project_id}|${data.board_id}|${data.board_side}|${data.frame_id}`;
    const geometryKey = `${key}|${width}x${height}`;
    if (state.imageGeometryKey !== geometryKey) {
      canvas.setAttribute("viewBox", `0 0 ${width} ${height}`);
      image.setAttribute("x","0"); image.setAttribute("y","0");
      image.setAttribute("width",String(width)); image.setAttribute("height",String(height));
      state.imageGeometryKey = geometryKey;
    }
    const readyOnceKey = key;
    const reportReady = () => {
      if (state.assetKey !== key || state.mountedAssetKey === key) return;
      const [expectedWidth, expectedHeight] = data.image_size_px;
      if (!image.isConnected || image.width.baseVal.value !== expectedWidth
          || image.height.baseVal.value !== expectedHeight) return;
      state.mountedAssetKey=key;
      if (data.viewer_ready_ack) return;
      setStateValue("viewer_status", { asset_key:key, status:"READY",
        image_size_px:[expectedWidth,expectedHeight], event_id:`ready-${Date.now()}` });
      telemetry("VIEWER_READY", {reason:"image decoded", ready_key_hash:readyOnceKey.slice(0,24)});
    };
    if (state.assetKey !== key) {
      const previousAssetKey = state.assetKey;
      telemetry("ASSET_KEY_CHANGED", {reason:previousAssetKey ? "image fingerprint changed" : "initial image"});
      state.assetKey=key;
      state.hover=null; state.drag=null; state.boxDraw=null; state.drawMode=false;
      state.selectedComponentIds=[];
      state.selectedComponentId=undefined; state.pendingSelection=null;
      tooltip.hidden=true; viewport.dataset.dragging="false"; viewport.dataset.drawing="false";
      const drawButton=root.querySelector('[data-action="draw-box"]');
      if(drawButton)drawButton.textContent="박스 그리기";
      state.mountedAssetKey="";
      image.addEventListener("load", reportReady, { once:true });
      image.setAttribute("href", `data:image/png;base64,${data.image_png_base64}`);
      state.hrefChangeCount += 1;
      telemetry("IMAGE_HREF_CHANGED", {reason:previousAssetKey ? "new image fingerprint" : "initial image"});
      telemetry("CAMERA_RESET", {reason:"image asset changed"});
      fit();
    }
    if (state.mountedAssetKey !== key) {
      if (typeof image.decode === "function") image.decode().then(reportReady).catch(() => {});
      if (image.complete) reportReady();
    }
  };
  if (state.listeners.length === 0) {
    // Keep Safari/Chromium from showing its blue text/image drag selection while
    // the pointer is used for selecting, panning, or drawing PCB boxes.
    listen(root,"dragstart",(event)=>event.preventDefault());
    listen(viewport,"wheel",(event)=>{ event.preventDefault(); zoomAt(event.deltaY<0?1.12:1/1.12,point(event.clientX,event.clientY)); },{passive:false});
    listen(viewport,"contextmenu",(event)=>{
      if(event.ctrlKey && event.target.closest("[data-component-id]"))event.preventDefault();
    });
    listen(viewport,"pointerdown",(event)=>{
      if (state.data.analysis_loading) return;
      if (event.button!==0) return;
      const p=point(event.clientX,event.clientY);
      if(state.drawMode){
        const x=(p.x-state.x)/state.scale,y=(p.y-state.y)/state.scale;
        state.boxDraw={id:event.pointerId,start:{x,y},end:{x,y}};
        viewport.dataset.drawing="true"; tooltip.hidden=true;
        try{viewport.setPointerCapture(event.pointerId)}catch(_){state.boxDraw=null}
        return;
      }
      state.drag={id:event.pointerId,startClientX:event.clientX,startClientY:event.clientY,
        x:p.x,y:p.y,tx:state.x,ty:state.y,moved:false};
    });
    listen(viewport,"pointermove",(event)=>{
      if(state.boxDraw && state.boxDraw.id===event.pointerId){
        const p=point(event.clientX,event.clientY),x=(p.x-state.x)/state.scale,y=(p.y-state.y)/state.scale;
        state.boxDraw.end={x,y};
        let preview=layer.querySelector('[data-draw-preview]');
        if(!preview){preview=svgElement("rect",{"data-draw-preview":"true",class:"cc-pcb-component cc-draw-preview"});layer.appendChild(preview)}
        const a=state.boxDraw.start,b=state.boxDraw.end;
        preview.setAttribute("x",String(Math.min(a.x,b.x)));preview.setAttribute("y",String(Math.min(a.y,b.y)));
        preview.setAttribute("width",String(Math.abs(b.x-a.x)));preview.setAttribute("height",String(Math.abs(b.y-a.y)));
        return;
      }
      const hit=hitComponent(event);
      const id=hit?.component_id || null;
      if (id!==state.hover) {
        state.hover=id;
        for (const shape of layer.querySelectorAll("[data-component-id]")) shape.classList.toggle("cc-hover",shape.dataset.componentId===id);
        if (hit) showTooltip(hit,event); else tooltip.hidden=true;
      } else if (hit) showTooltip(hit,event);
      if (!state.drag || state.drag.id!==event.pointerId) return;
      const dx=event.clientX-state.drag.startClientX, dy=event.clientY-state.drag.startClientY;
      if (!state.drag.moved && Math.hypot(dx,dy)<DRAG_THRESHOLD_PX) return;
      if (!state.drag.moved) {
        state.drag.moved=true; viewport.dataset.dragging="true";
        try { viewport.setPointerCapture(event.pointerId); } catch (_) { /* continue panning without capture */ }
      }
      const p=point(event.clientX,event.clientY); state.x=state.drag.tx+p.x-state.drag.x; state.y=state.drag.ty+p.y-state.drag.y;
      state.userAdjusted=true; drawCamera();
    });
    listen(viewport,"pointerleave",()=>{ tooltip.hidden=true; state.hover=null; });
    listen(viewport,"pointerup",(event)=>{
      if(state.boxDraw && state.boxDraw.id===event.pointerId){
        const {start,end}=state.boxDraw; state.boxDraw=null;
        const [width,height]=state.data.image_size_px;
        const bbox=[Math.max(0,Math.min(width,Math.min(start.x,end.x))),
          Math.max(0,Math.min(height,Math.min(start.y,end.y))),
          Math.max(0,Math.min(width,Math.max(start.x,end.x))),
          Math.max(0,Math.min(height,Math.max(start.y,end.y)))];
        layer.querySelector('[data-draw-preview]')?.remove();
        viewport.dataset.drawing="false";state.drawMode=false;
        const drawButton=root.querySelector('[data-action="draw-box"]');
        if(drawButton)drawButton.textContent="박스 그리기";
        if(bbox[2]-bbox[0]>=4&&bbox[3]-bbox[1]>=4)sendSelection(null,"draw",bbox);
        try{if(viewport.hasPointerCapture(event.pointerId))viewport.releasePointerCapture(event.pointerId)}catch(_){/* already released */}
        return;
      }
      const drag=state.drag;
      if (!drag || drag.id!==event.pointerId) return;
      state.drag=null;
      if (!drag.moved) {
        // Safari may retarget or omit the synthesized click after pointer
        // capture. The pointerup coordinates are still the physical click.
        const hits=hitComponents(event).filter((item)=>item.overlay_status!=="EXCLUDED");
        const ids=hits.map((item)=>item.component_id);
        let selectedId=ids[0] || null;
        console.debug("[pcb-selection] pointerup_select", {x:event.clientX,y:event.clientY,
          selected_component_id:selectedId, overlapping_component_ids:ids,
          asset_key:state.assetKey});
        if ((event.metaKey || event.ctrlKey) && selectedId) {
          const next=new Set(state.selectedComponentIds||[]);
          if(next.has(selectedId))next.delete(selectedId);else next.add(selectedId);
          state.selectedComponentIds=[...next];
          selectedId=next.has(ids[0])?ids[0]:(state.selectedComponentIds[state.selectedComponentIds.length-1]||null);
          for(const [id,entry] of state.regionNodes||[])
            entry.shape.classList.toggle("cc-bulk-selected",next.has(id));
          applySelectedHighlight(selectedId);
          sendSelection(selectedId,next.size?"bulk-select":"clear",null,[...next]);
        } else {
          state.selectedComponentIds=[];
          for(const [id,entry] of state.regionNodes||[])entry.shape.classList.remove("cc-bulk-selected");
          applySelectedHighlight(selectedId);
          sendSelection(selectedId, null, null, ids);
        }
      }
      viewport.dataset.dragging="false";
      try { if(viewport.hasPointerCapture(event.pointerId))viewport.releasePointerCapture(event.pointerId); } catch (_) { /* already released */ }
    });
    const cancelDrag=(event)=>{
      if(state.boxDraw && state.boxDraw.id===event.pointerId){
        state.boxDraw=null;layer.querySelector('[data-draw-preview]')?.remove();
        viewport.dataset.drawing="false";return;
      }
      if (!state.drag || state.drag.id!==event.pointerId) return;
      state.drag=null; viewport.dataset.dragging="false"; tooltip.hidden=true;
      try { if(viewport.hasPointerCapture(event.pointerId))viewport.releasePointerCapture(event.pointerId); } catch (_) { /* already released */ }
    };
    listen(viewport,"pointercancel",cancelDrag); listen(viewport,"lostpointercapture",cancelDrag);
    listen(layer,"keydown",(event)=>{ if(event.key!=="Enter"&&event.key!==" ")return; const shape=event.target.closest("[data-component-id]"); if(!shape)return; event.preventDefault(); const id=shape.dataset.componentId; applySelectedHighlight(id); sendSelection(id, null, null, [id]); });
    for(const button of root.querySelectorAll("[data-action]")) listen(button,"click",()=>{
      const action=button.dataset.action; if(action==="clear-selection"){
        state.selectedComponentIds=[];
        for(const [id,entry] of state.regionNodes||[])entry.shape.classList.remove("cc-bulk-selected");
        applySelectedHighlight(null);sendSelection(null);
      }
      else if(action==="draw-box"){
        state.drawMode=!state.drawMode;viewport.dataset.drawing=state.drawMode?"true":"false";
        button.textContent=state.drawMode?"그리기 취소":"박스 그리기";
      }
      else if(action==="fit"||action==="reset")fit();
      else { const r=canvas.getBoundingClientRect(); zoomAt(action==="zoom-in"?1.25:0.8,point(r.left+r.width/2,r.top+r.height/2)); }
    });
    state.observer=new ResizeObserver(()=>{
      tooltip.hidden=true; state.hover=null;
      if(!state.userAdjusted)fit();
    }); state.observer.observe(viewport);
  }
  const previousOverlayVersion=state.overlayVersion;
  setImage(); renderRegions(); drawCamera();
  if (previousOverlayVersion !== data.overlay_version) {
    telemetry("REGION_PATCH", {reason:previousOverlayVersion ? "overlay version changed" : "initial regions",
      overlay_version:data.overlay_version, component_count:(data.components || []).length});
    state.overlayVersion=data.overlay_version;
  }
  return ()=>{
    for(const remove of state.listeners.splice(0))remove();
    state.observer?.disconnect();
    if (!root.isConnected && state.mounted) { telemetry("VIEWER_UNMOUNT", {reason:"component DOM removed"}); state.mounted=false; }
  };
}
