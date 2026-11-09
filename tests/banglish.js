// Banglish questions become Bangla before the crop, pest and answer checks; English and product names stay as typed.
// Run: python -m http.server 8765 -d docs  then  node tests/banglish.js
const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();const e=[];p.on('pageerror',x=>e.push(x.message));await p.route(/jsdelivr/,r=>r.abort());
await p.goto('http://localhost:8765/');await p.waitForTimeout(800);
const cases=[["amar dhaner pata holud hoye jacche keno","আমার ধানের পাতা হলুদ হয়ে যাচ্ছে কেন"],["boro dhan kokhon lagabo?","বোরো ধান কখন লাগাব?"],
 ["tomato gacher pata kukre jacche ki korbo","টমেটো গাছের পাতা কুঁকড়ে যাচ্ছে কী করব"],["Begun e poka dhorse, osud ki dimu","বেগুনে পোকা ধরছে, ওষুধ কী দিব"],
 ["kemon achen","কেমন আছেন"],["what is the dose of tomato blight spray","what is the dose of tomato blight spray"],["hi","hi"],["dhonnobad vai","ধন্যবাদ ভাই"]];
const got=await p.evaluate(c=>c.map(([q])=>fromBanglish(q)),cases);
const crops=await p.evaluate(()=>[cropsIn(fromBanglish("amar aloo te morok lagse")).map(c=>c.k), !!pestIn(fromBanglish("aloo te nabi dhosa")), smallTalk(fromBanglish("kemon achen")), smallTalk(fromBanglish("Tor ki obostha")) && /ভালো আছি/.test(answer(fromBanglish("Tor ki obostha"))), smallTalk(fromBanglish("tumi ke"))]);
let ok=e.length===0; cases.forEach(([q,want],i)=>{ if(got[i]!==want){ok=false;console.log("MISS",q,"=>",got[i]);} });
if(crops[0][0]!=="potato"||!crops[1]||!crops[2]||!crops[3]||!crops[4]){ok=false;console.log("checks",crops);}
console.log(ok?"banglish ok":"banglish FAILED", e); process.exitCode=ok?0:1; await b.close();})();
// everyday English words are not product names: "message" must not open the Message 2.5EC card
(async()=>{const {chromium}=require('playwright');const b=await chromium.launch();const p=await b.newPage();await p.route(/jsdelivr|hf.space|google/,r=>r.abort());await p.goto('http://localhost:8765/');await p.waitForTimeout(1500);
const r=await p.evaluate(()=>[brandIn(fromBanglish("bhalo kore por ager message ki bolsi")), brandIn("Message 2.5EC ধানে কত দিব"), fromBanglish("Shifat ekjon farmer,ore kisu tips deo")]);
const ok=r[0]===null&&r[1]==="Message 2.5EC"&&r[2]==="Shifat একজন farmer,ওকে কিছু tips দাও"; console.log(ok?"brand ok":"brand FAILED",r); if(!ok)process.exitCode=1; await b.close();})();
