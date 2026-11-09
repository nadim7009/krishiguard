// Proportion questions are worked out on the page. Run: python -m http.server 8765 -d docs  then  node tests/calc.js
const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();const e=[];p.on('pageerror',x=>e.push(x.message));await p.route(/jsdelivr|hf.space|google/,r=>r.abort());
await p.goto('http://localhost:8765/');await p.waitForTimeout(800);
const cases=[["ধানের জমিতে ১০ লিটার পানিতে ২০ মিলি কীটনাশক মেশাতে বলা হয়েছে। ১৫ লিটার পানি নিলে কত মিলি লাগবে?","৩০ মিলি"],
 ["এক একর জমিতে ৫০ কেজি সার লাগে। ৩০ শতক জমির জন্য কত কেজি লাগবে?","১৫ কেজি"],["১ বিঘায় ২০ কেজি ইউরিয়া, ৫ বিঘায় কত কেজি?","১০০ কেজি"],
 ["প্রতি লিটারে ২ মিলি, ১৬ লিটার ট্যাংকে কত মিলি?","৩২ মিলি"],["৩৩ শতকে ১০ কেজি হলে ১ একরে কত গ্রাম?","৩০৩০৩.০৩ গ্রাম"],["ধানের পাতা হলুদ কেন?",null]];
const got=await p.evaluate(c=>c.map(([q])=>{const h=ratioCalc(q);return h&&h.replace(/<[^>]+>/g,"");}),cases);
let ok=!e.length; cases.forEach(([q,w],i)=>{const g=got[i]; const pass= w===null? g===null : !!g&&g.includes(w); if(!pass){ok=false;} console.log(pass?"ok  ":"MISS",q.slice(0,40),"=>",g&&g.slice(0,120));});
console.log(ok?"calc ok":"calc FAILED",e);process.exitCode=ok?0:1;await b.close();})();
// general card: run after the main check
(async()=>{const {chromium}=require('playwright');const b=await chromium.launch();const p=await b.newPage();await p.route(/jsdelivr|hf.space|google/,r=>r.abort());await p.goto('http://localhost:8765/');await p.waitForTimeout(800);
const g=await p.evaluate(()=>[!!generalCard("ফসলের পোকা দমনে কীভাবে নিরাপদ পদ্ধতি ব্যবহার করা যায়?"),!!generalCard("জৈব উপায়ে পোকা দমন"),!generalCard("ধানের পাতা হলুদ কেন")]);
console.log(g.every(Boolean)?"general ok":"general FAILED",g);if(!g.every(Boolean))process.exitCode=1;await b.close();})();
