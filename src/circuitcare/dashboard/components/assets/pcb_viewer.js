const SVG_NS = "http://www.w3.org/2000/svg";
const DRAG_THRESHOLD_PX = 5;
const instances = new WeakMap();

function svgElement(name, attrs = {}) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  return node;
}

export default function(component) {
  const { parentElement, data, setStateValue } = component;
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
      listeners:[], observer:null, sequence:0, userAdjusted:false, mountedAssetKey:"" };
    instances.set(root, state);
  }
  state.data = data;
  state.setStateValue = setStateValue;
  state.sequence = Math.max(state.sequence, Number(data.event_sequence_seed) || 0);
  const listen = (target, event, callback, options) => {
    target.addEventListener(event, callback, options);
    state.listeners.push(() => target.removeEventListener(event, callback, options));
  };
  const point = (clientX, clientY) => {
    const p = canvas.createSVGPoint(); p.x = clientX; p.y = clientY;
    return p.matrixTransform(canvas.getScreenCTM().inverse());
  };
  const hitComponent = (event) => {
    const p = point(event.clientX, event.clientY);
    const x = (p.x - state.x) / state.scale;
    const y = (p.y - state.y) / state.scale;
    return (state.data.components || []).filter((item) => {
      const [x1, y1, x2, y2] = item.bbox_xyxy;
      return x >= x1 && x <= x2 && y >= y1 && y <= y2;
    }).sort((a, b) => {
      const aSelected = a.component_id === state.data.selected_component_id;
      const bSelected = b.component_id === state.data.selected_component_id;
      if (aSelected !== bSelected) return aSelected ? -1 : 1;
      const aArea = (a.bbox_xyxy[2] - a.bbox_xyxy[0]) * (a.bbox_xyxy[3] - a.bbox_xyxy[1]);
      const bArea = (b.bbox_xyxy[2] - b.bbox_xyxy[0]) * (b.bbox_xyxy[3] - b.bbox_xyxy[1]);
      return aArea - bArea || a.component_id.localeCompare(b.component_id);
    })[0] || null;
  };
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
  const sendSelection = (componentId, action = null, bbox = null) => {
    state.sequence += 1;
    state.setStateValue("selection", {
      project_id:state.data.project_id, board_id:state.data.board_id, board_side:state.data.board_side,
      component_id:componentId, canonical_frame_id:state.data.frame_id,
      selection_action:action || (componentId ? "select" : "clear"), bbox_xyxy:bbox, event_seq:state.sequence,
      event_id:`${Date.now()}-${state.sequence}`,
    });
  };
  const showTooltip = (item, event) => {
    const typeLabels = { capacitor:"커패시터", resistor:"저항", inductor:"인덕터", diode:"다이오드",
      transistor:"트랜지스터", integrated_circuit:"IC", ic:"IC", connector:"커넥터",
      unknown_component:"부품 후보" };
    const rows = [typeLabels[item.component_type?.toLowerCase()] || item.component_type];
    if (item.display_label) rows.push(`Reference ${item.display_label}`);
    rows.push(`식별 ${item.display_status || "미확인"}`);
    if (item.ocr_marking) rows.push(`표기 ${item.ocr_marking}`);
    const faultLabels = { SUSPECTED_FAULT:"고장 의심 후보", RELATED_COMPONENT:"관련 점검 후보",
      CONFIRMED_FAULT:"검사 근거 확인" };
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
    tooltip.textContent = rows.join("\n"); tooltip.hidden = false;
    const rect = viewport.getBoundingClientRect();
    tooltip.style.left = `${Math.min(Math.max(8, event.clientX - rect.left + 12), rect.width - 220)}px`;
    tooltip.style.top = `${Math.max(8, event.clientY - rect.top + 12)}px`;
  };
  const renderRegions = () => {
    layer.replaceChildren();
    const fragment = document.createDocumentFragment();
    for (const item of data.components || []) {
      const [x1,y1,x2,y2] = item.bbox_xyxy;
      const shape = svgElement("rect", { x:x1, y:y1, width:x2-x1, height:y2-y1,
        class:`cc-pcb-component${item.component_id === data.selected_component_id ? " cc-selected" : ""}${item.overlay_status === "EXCLUDED" ? " cc-excluded" : ""}${item.fault_candidate_status === "SUSPECTED_FAULT" ? " cc-fault-suspected" : item.fault_candidate_status === "RELATED_COMPONENT" ? " cc-fault-related" : item.fault_candidate_status === "CONFIRMED_FAULT" ? " cc-fault-confirmed" : ""}`,
        "data-component-id":item.component_id, tabindex:"0", role:"button",
        "aria-label":`${item.display_label || item.component_type} 부품 선택` });
      fragment.appendChild(shape);
      if (item.display_label) {
        const label = svgElement("text", { x:(x1+x2)/2, y:Math.max(12,y1-4), class:"cc-pcb-label" });
        label.textContent = item.display_label; fragment.appendChild(label);
      }
    }
    layer.appendChild(fragment);
  };
  const setImage = () => {
    const [width,height] = data.image_size_px || [];
    if (!data.image_png_base64 || !width || !height) { empty.hidden=false; return; }
    empty.hidden=true;
    canvas.setAttribute("viewBox", `0 0 ${width} ${height}`);
    image.setAttribute("x","0"); image.setAttribute("y","0");
    image.setAttribute("width",String(width)); image.setAttribute("height",String(height));
    const key = `${data.asset_key}|${data.project_id}|${data.board_id}|${data.board_side}|${data.frame_id}`;
    const reportReady = () => {
      if (state.assetKey !== key || state.mountedAssetKey === key) return;
      const [expectedWidth, expectedHeight] = data.image_size_px;
      if (!image.isConnected || image.width.baseVal.value !== expectedWidth
          || image.height.baseVal.value !== expectedHeight) return;
      state.mountedAssetKey=key;
      setStateValue("viewer_status", { asset_key:key, status:"READY",
        image_size_px:[expectedWidth,expectedHeight], event_id:`ready-${Date.now()}` });
    };
    if (state.assetKey !== key) {
      state.assetKey=key;
      state.hover=null; state.drag=null; state.boxDraw=null; state.drawMode=false;
      tooltip.hidden=true; viewport.dataset.dragging="false"; viewport.dataset.drawing="false";
      const drawButton=root.querySelector('[data-action="draw-box"]');
      if(drawButton)drawButton.textContent="박스 그리기";
      state.mountedAssetKey="";
      image.addEventListener("load", reportReady, { once:true });
      image.setAttribute("href", `data:image/png;base64,${data.image_png_base64}`);
      fit();
    }
    if (state.mountedAssetKey !== key) {
      if (typeof image.decode === "function") image.decode().then(reportReady).catch(() => {});
      if (image.complete) reportReady();
    }
  };
  if (state.listeners.length === 0) {
    listen(viewport,"wheel",(event)=>{ event.preventDefault(); zoomAt(event.deltaY<0?1.12:1/1.12,point(event.clientX,event.clientY)); },{passive:false});
    listen(viewport,"pointerdown",(event)=>{
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
      try { viewport.setPointerCapture(event.pointerId); } catch (_) { state.drag=null; return; }
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
      state.drag.moved=true; viewport.dataset.dragging="true";
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
        const hit=hitComponent(event);
        sendSelection(hit?.component_id || null);
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
    listen(layer,"keydown",(event)=>{ if(event.key!=="Enter"&&event.key!==" ")return; const shape=event.target.closest("[data-component-id]"); if(!shape)return; event.preventDefault(); sendSelection(shape.dataset.componentId); });
    for(const button of root.querySelectorAll("[data-action]")) listen(button,"click",()=>{
      const action=button.dataset.action; if(action==="clear-selection")sendSelection(null);
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
  setImage(); renderRegions(); drawCamera();
  return ()=>{ for(const remove of state.listeners.splice(0))remove(); state.observer?.disconnect(); };
}
