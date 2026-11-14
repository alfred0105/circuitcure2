const instances = new WeakMap();
export default function(component) {
  const {parentElement, data, setTriggerValue} = component;
  const root = parentElement.querySelector('[data-review-progress]');
  let state = instances.get(root);
  if (!state) { state={job:null, notified:false}; instances.set(root,state); }
  if (state.job !== data.job_id) {state.job=data.job_id;state.notified=false;}
  const stage=root.querySelector('[data-stage]'), elapsed=root.querySelector('[data-elapsed]');
  const bar=root.querySelector('progress'), detail=root.querySelector('[data-detail]');
  const labels={PENDING:'AI 검수 대기 중',OCR_BOARD:'RapidOCR · PCB 문자 탐색',OCR_MARKINGS:'RapidOCR · 소자 각인',QWEN_CLASS_REVIEW:'Qwen3.5 · 소자 분류',QWEN_DUPLICATE_REVIEW:'Qwen3.5 · 중복 검토',RESULT_VALIDATION:'결과 검증',COMPLETE:'검수 완료',FAILED:'검수 실패'};
  let base=0, received=Date.now(), terminal=false;
  const tick=()=>{elapsed.textContent=`${Math.floor((base+(terminal?0:Date.now()-received))/1000)}초`;};
  const update=(job)=>{
    const p=job.progress || {};
    base=Number(job.elapsed_ms)||base;received=Date.now();
    terminal=!['PENDING','RUNNING'].includes(job.status);
    stage.textContent=labels[p.stage] || job.status || 'AI 검수 중';
    const total=Number(p.total_batches)||0, completed=Number(p.completed_batches)||0;
    bar.max=total||1;bar.value=terminal&&job.status==='COMPLETE'?bar.max:Math.min(completed,bar.max);
    detail.textContent=job.error || p.message || (terminal?'결과를 반영합니다.':'AI 응답을 기다리고 있습니다.');
    tick();
    if(terminal&&!state.notified){
      state.notified=true;
      setTriggerValue('completed',{job_id:data.job_id,token:data.token});
    }
  };
  update(data.initial);
  const stream=terminal?null:new EventSource(data.url);
  if(stream){
    stream.onmessage=(event)=>{try{const job=JSON.parse(event.data);update(job);if(terminal)stream.close();}catch(_){detail.textContent='진행 상태 응답을 확인하지 못했습니다.';}};
    stream.onerror=()=>{if(!terminal)detail.textContent='진행 상태 연결을 복구하고 있습니다. AI 작업은 계속됩니다.';};
  }
  const timer=setInterval(tick,1000);
  return ()=>{clearInterval(timer);stream?.close();};
}
