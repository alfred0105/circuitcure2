from pathlib import Path
import shutil
import subprocess
import pytest


def test_progress_ticks_do_not_trigger_python_and_completion_is_once(tmp_path):
    node=shutil.which('node')
    if not node:
        pytest.skip('Node unavailable')
    source=Path(__file__).parents[1]/'src/circuitcare/dashboard/components/assets/review_progress.js'
    script=tmp_path/'progress.cjs'
    script.write_text(r'''
const fs=require('fs'), vm=require('vm'), assert=require('assert');
const nodes={}, root={querySelector:k=>nodes[k] ||= {textContent:'',max:1,value:0}};
let ticks, cleared=false, events=[], streams=[];
class EventSource{constructor(url){this.url=url;streams.push(this);}close(){this.closed=true;}}
const context={EventSource,Date,setInterval:fn=>{ticks=fn;return 1;},clearInterval:()=>cleared=true};
vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[2],'utf8').replace('export default function(component)', 'globalThis.render=function(component)'),context);
const data={job_id:'j1',token:'capability',url:'http://localhost/events/token',
 initial:{status:'RUNNING',elapsed_ms:1000,progress:{stage:'OCR_BOARD',message:'OCR'}}};
const cleanup=context.render({parentElement:{querySelector:()=>root},data,setTriggerValue:(k,v)=>events.push({k,v})});
for(let i=0;i<30;i++)ticks();
assert.equal(events.length,0);assert.equal(streams.length,1);
streams[0].onmessage({data:JSON.stringify({status:'RUNNING',progress:{stage:'QWEN_CLASS_REVIEW',total_batches:6,completed_batches:2}})});
assert.equal(nodes.progress.value,2);assert.equal(events.length,0);
streams[0].onerror();assert.equal(events.length,0);
const done={data:JSON.stringify({status:'COMPLETE',progress:{stage:'COMPLETE'}})};
streams[0].onmessage(done);streams[0].onmessage(done);
assert.equal(events.length,1);assert.equal(events[0].v.job_id,'j1');assert.equal(events[0].v.token,'capability');
assert.equal(streams[0].closed,true);cleanup();assert.equal(cleared,true);
''')
    subprocess.run([node,str(script),str(source)],check=True,capture_output=True,text=True)
