// Whole voice mode on a phone-sized page: fake microphone (speech inside room noise), model server mocked.
// The browser recognizer is unavailable here, so this goes through the recorder path. Expect: listen -> think -> answer -> talk -> listen again.
// Run: python -m http.server 8765 -d docs  then  node tests/voice_mode.js
const {chromium, devices}=require('playwright'), fs=require('fs'), os=require('os'), path=require('path');
const sr=16000; let seed=7; const rnd=()=>{seed=(seed*16807)%2147483647;return seed/2147483647;}; const gauss=()=>Math.sqrt(-2*Math.log(rnd()+1e-9))*Math.cos(2*Math.PI*rnd());
function wav(file, secs, segs){ const n=sr*secs, d=Buffer.alloc(44+n*2);
  d.write('RIFF',0); d.writeUInt32LE(36+n*2,4); d.write('WAVEfmt ',8); d.writeUInt32LE(16,16); d.writeUInt16LE(1,20); d.writeUInt16LE(1,22); d.writeUInt32LE(sr,24); d.writeUInt32LE(sr*2,28); d.writeUInt16LE(2,32); d.writeUInt16LE(16,34); d.write('data',36); d.writeUInt32LE(n*2,40);
  for(let i=0;i<n;i++){ const t=i/sr; let x=0.03*gauss(); if(segs.some(([a,b])=>t>=a&&t<b)) x+=0.25*Math.sin(2*Math.PI*220*t)*(0.6+0.4*Math.sin(2*Math.PI*3*t)); d.writeInt16LE(Math.max(-32767,Math.min(32767,Math.round(x*32767))),44+i*2);} fs.writeFileSync(file,d); }
const tiny=(()=>{const n=1600,d=Buffer.alloc(44+n*2);d.write('RIFF',0);d.writeUInt32LE(36+n*2,4);d.write('WAVEfmt ',8);d.writeUInt32LE(16,16);d.writeUInt16LE(1,20);d.writeUInt16LE(1,22);d.writeUInt32LE(16000,24);d.writeUInt32LE(32000,28);d.writeUInt16LE(2,32);d.writeUInt16LE(16,34);d.write('data',36);d.writeUInt32LE(n*2,40);return d;})();
(async()=>{ const f=path.join(fs.mkdtempSync(path.join(os.tmpdir(),'vm-')),'talk.wav'); wav(f,60,[[1,3],[20,22]]);
 const b=await chromium.launch({args:['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-audio-capture='+f,'--autoplay-policy=no-user-gesture-required']});
 const ctx=await b.newContext({...devices['Pixel 7'], permissions:['microphone']}); const p=await ctx.newPage(); const log=[];
 p.on('pageerror',e=>log.push('PAGEERROR '+e.message)); p.on('console',m=>{ if(m.type()==='error') log.push('console '+m.text().slice(0,120)); });
 await p.route(/jsdelivr|google|supabase/,r=>r.abort());
 await p.route(/hf\.space/, async r=>{ const u=r.request().url(), body=r.request().postData()||'';
   if(u.endsWith('/tts')) return r.fulfill({status:200,contentType:'audio/wav',body:tiny});
   if(u.endsWith('/search')) return r.fulfill({status:200,contentType:'application/json',body:'{"results":[]}'});
   if(body.includes('input_audio')) return r.fulfill({status:200,contentType:'application/json',body:JSON.stringify({choices:[{message:{content:'কেমন আছেন'}}]})});
   if(body.includes('"stream":true')) return r.fulfill({status:200,contentType:'text/event-stream',body:'data: '+JSON.stringify({choices:[{delta:{content:'আমি ভালো আছি।'}}]})+'\n\ndata: [DONE]\n\n'});
   return r.fulfill({status:200,contentType:'application/json',body:'{}'}); });
 await p.goto('http://localhost:8765/#chat'); await p.waitForTimeout(1500);
 await p.evaluate(()=>{ Object.defineProperty(cloud,'guest',{get:()=>false}); window.STATES=[]; const ov=document.getElementById('vmode');
   new MutationObserver(()=>STATES.push(Math.round(performance.now())+':'+ov.dataset.state)).observe(ov,{attributes:true,attributeFilter:['data-state']}); window.T0=performance.now(); });
 await p.evaluate(()=>{ const go=()=>{ const g=document.getElementById('authGate')||document.querySelector('.auth'); }; go(); document.getElementById('voiceBtn').click(); });
 await p.waitForTimeout(16000);
 const r=await p.evaluate(()=>({states:STATES.map(s=>{const [t,k]=s.split(':');return Math.round((t-T0)/100)/10+'s '+k;}), heard:document.getElementById('vmHeard').textContent, hidden:document.getElementById('vmode').hidden, msgs:history.map(m=>m.role+':'+(m.text||m.html||'').replace(/<[^>]+>/g,'').slice(0,40))}));
 console.log(JSON.stringify(r,null,1)); console.log(log.join('\n'));
 const ok=r.states.some(s=>s.endsWith('think'))&&r.states.some(s=>s.endsWith('talk'))&&r.states.filter(s=>s.endsWith('listen')).length>=2;
 console.log(ok?'voice mode ok':'voice mode FAILED'); process.exitCode=ok?0:1; await b.close(); })();
