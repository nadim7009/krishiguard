// The recorder stops by itself in a noisy room: speech inside steady noise ends soon after the words, noise alone ends at 7 s with nothing.
// Run: python -m http.server 8765 -d docs  then  node tests/voice_vad.js
const {chromium}=require('playwright'), fs=require('fs'), os=require('os'), path=require('path');
const sr=16000; let seed=1; const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;}; const gauss=()=>Math.sqrt(-2*Math.log(rnd()+1e-9))*Math.cos(2*Math.PI*rnd());
function wav(file, secs, speechFrom, speechTo){ const n=sr*secs, d=Buffer.alloc(44+n*2);
  d.write('RIFF',0); d.writeUInt32LE(36+n*2,4); d.write('WAVEfmt ',8); d.writeUInt32LE(16,16); d.writeUInt16LE(1,20); d.writeUInt16LE(1,22); d.writeUInt32LE(sr,24); d.writeUInt32LE(sr*2,28); d.writeUInt16LE(2,32); d.writeUInt16LE(16,34); d.write('data',36); d.writeUInt32LE(n*2,40);
  for(let i=0;i<n;i++){ const t=i/sr; let x=0.03*gauss(); if(t>=speechFrom&&t<speechTo) x+=0.25*Math.sin(2*Math.PI*220*t)*(0.6+0.4*Math.sin(2*Math.PI*3*t)); d.writeInt16LE(Math.max(-32767,Math.min(32767,Math.round(x*32767))),44+i*2);} fs.writeFileSync(file,d); }
(async()=>{ const dir=fs.mkdtempSync(path.join(os.tmpdir(),'vad-')); let ok=true;
 for (const [name, from, to, check] of [['noisy_speech',1,3,r=>/done@(\d+):আমার/.test(r.join())&&+r.join().match(/done@(\d+)/)[1]<6000], ['noise_only',99,99,r=>/^done@(\d+):$/.test(r[r.length-1])], ['talks_at_once',0,2.5,r=>/done@(\d+):আমার/.test(r.join())], ['short_bang',2,2.2,r=>/^done@(\d+):$/.test(r[r.length-1])]]){
  const f=path.join(dir,name+'.wav'); wav(f,20,from,to);
  const b=await chromium.launch({args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+f]});
  const p=await (await b.newContext({permissions:['microphone']})).newPage(); await p.route(/jsdelivr|google/,r=>r.abort());
  await p.route(/hf\.space/,r=>r.fulfill({status:200,contentType:'application/json',body:JSON.stringify({choices:[{message:{content:'আমার ধানের পাতা হলুদ'}}]})}));
  await p.goto('http://localhost:8765/'); await p.waitForTimeout(800);
  const r=await p.evaluate(()=>new Promise(res=>{SR_OK=false; const t0=performance.now(), log=[];
    recordVoice({interim(){}, busy:()=>log.push('busy'), done:t=>res([...log,'done@'+Math.round(performance.now()-t0)+':'+t]), fail:w=>res([...log,'fail:'+w])}, {stop(){}});
    setTimeout(()=>res([...log,'still listening']),20000);}));
  const pass=check(r); ok=ok&&pass; console.log(pass?'ok  ':'FAIL',name,r); await b.close(); }
 console.log(ok?'voice ok':'voice FAILED'); process.exitCode=ok?0:1; })();
