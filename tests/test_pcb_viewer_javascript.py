"""Execute Viewer event delivery in Node; not a browser geometry/E2E test."""
import shutil
import subprocess
from pathlib import Path

import pytest


def test_viewer_click_and_keyboard_emit_triggers_after_data_update(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("Node.js unavailable")
    source = Path(__file__).parents[1] / "src/circuitcare/dashboard/components/assets/pcb_viewer.js"
    script = tmp_path / "viewer-test.cjs"
    script.write_text(r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
class Element {
 constructor(){this.dataset={};this.events={};this.classList={add(){},remove(){},toggle(){}};this.clientWidth=800;this.clientHeight=600;}
 setAttribute(){} append(){} remove(){} setPointerCapture(){} hasPointerCapture(){return false;} releasePointerCapture(){} getBoundingClientRect(){return {left:0,top:0,width:800,height:600};}
 addEventListener(name,cb){this.events[name]=cb;} removeEventListener(name){delete this.events[name];}
 querySelectorAll(){return [];} querySelector(){return null;}
 createSVGPoint(){return {x:0,y:0,matrixTransform(){return {x:15,y:15};}};}
 getScreenCTM(){return {inverse(){return {};}};}
}
const root=new Element(), nodes={};
root.querySelector=(key)=>nodes[key] ||= new Element();
const parent={querySelector(){return root;}};
const context={console:{debug(){}},document:{createElementNS(){return new Element();}},
 ResizeObserver:class{observe(){} disconnect(){}},setTimeout};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('export default function(component)', 'globalThis.render = function(component)'),context);
const data={components:[{component_id:'r1',bbox_xyxy:[10,10,20,20],component_type:'resistor'}],
 image_size_px:[100,100],overlay_version:'v1',project_id:'p',board_id:'b',frame_id:'photo',event_epoch:'epoch'};
 let first=[],second=[];
 const cleanup=context.render({parentElement:parent,data,setStateValue:(key,value)=>first.push({key,value}),setTriggerValue(){}});
 nodes['[data-viewport]'].events.pointerdown({button:0,pointerId:1,clientX:15,clientY:15});
 nodes['[data-viewport]'].events.pointerup({pointerId:1,clientX:15,clientY:15});
assert.equal(first[0].key,'selection_event');assert.equal(first[0].value.component_id,'r1');
cleanup();
 context.render({parentElement:parent,data:{...data,selected_component_id:'r1'},setStateValue:(key,value)=>second.push({key,value}),
  setTriggerValue(){}});
nodes['[data-component-layer]'].events.keydown({key:'Enter',preventDefault(){},target:{closest(){return {dataset:{componentId:'r1'}};}}});
assert.equal(second.length,1);assert.equal(first.length,1);assert.equal(second[0].value.component_id,'r1');
assert.equal(second[0].value.event_epoch,'epoch');
 const hide=context.render({parentElement:parent,data:{...data,analysis_loading:true},setStateValue:(key,value)=>second.push({key,value}),
  setTriggerValue(){}});
assert.equal(root.dataset.loading,'true');
 nodes['[data-viewport]'].events.pointerdown({button:0,pointerId:2,clientX:15,clientY:15});
 nodes['[data-viewport]'].events.pointerup({pointerId:2,clientX:15,clientY:15});
assert.equal(second.length,1);
hide();
 context.render({parentElement:parent,data:{...data,analysis_loading:false},setStateValue:(key,value)=>second.push({key,value}),
  setTriggerValue(){}});
assert.equal(root.dataset.loading,'false');
 nodes['[data-viewport]'].events.pointerdown({button:0,pointerId:3,clientX:15,clientY:15});
 nodes['[data-viewport]'].events.pointerup({pointerId:3,clientX:15,clientY:15});
assert.equal(second.length,2);

''', encoding="utf-8")
    subprocess.run([node, str(script), str(source)], check=True, capture_output=True, text=True)
