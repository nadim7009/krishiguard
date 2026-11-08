// Checks the website answer guard: a made-up product (notebook 11, maize cutworm) is removed, a farmer typo echoed back is kept.
// Run: python -m http.server 8765 -d docs  then  node tests/site_guard.js
const {chromium}=require('playwright');(async()=>{const b=await chromium.launch();const p=await b.newPage();const e=[];p.on('pageerror',x=>e.push(x.message));await p.route(/jsdelivr/,r=>r.abort());
await p.goto('http://localhost:8765/');await p.waitForTimeout(500);
const r=await p.evaluate(()=>{
 const ev1='পণ্য Karate 2.5EC (Lambda Cyhalothrin · ভুট্টা, কাটুই পোকা): নিবন্ধিত মাত্রা 0.75 L/ha\nনিবন্ধন নম্বর: DAE AP-263';
 const t1='ভুট্টা ফসলে কাটুই পোকা দমনে নিবন্ধিত কয়েকটি বালাইনাশক:\n- Karate 2.5EC (Lambda Cyhalothrin): ০.৭৫ L/ha [DAE AP-263]\n- Dithane 2.5EC (Cypermethrin): ০.৭৫ L/ha [DAE AP-263]\n';
 const ev2='পণ্য Orozole 5SC (Hexaconazole · ধান, খোলপোড়া): নিবন্ধিত মাত্রা 500 ml/ha\nনিবন্ধন নম্বর: DAE AP-1020';
 const t2='ধান ফসলে খোলপোড়া রোগ দমনে Orzole 5SC (Hexaconazole) নিবন্ধিত। মাত্রা প্রতি হেক্টরে 500 ml।';
 const ev3='পণ্য Taba 25WDG (Ethion (5%) + Thiamethoxam (20%)): নিবন্ধিত মাত্রা 100 g/ha';
 const t3='ধান ফসলে কারেন্ট পোকা দমনে Taba 25WDG (Ethion (5%) + Thiamethoxam (20%)) নিবন্ধিত। নিবন্ধিত মাত্রা প্রতি হেক্টরে 100 g।';
 return [numberGuard(t1,ev1),numberGuard(t2,ev2),numberGuard(t3,ev3),numberGuard('Overall fine text','')];});
const ok=r[0].dropped===1&&!/Dithane/.test(r[0].text)&&r[1].dropped===0&&r[2].dropped===0&&e.length===0; console.log(ok?"guard ok":"guard FAILED",JSON.stringify(r)); process.exitCode=ok?0:1;await b.close();})();
