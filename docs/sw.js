// KrishiGuard offline cache: same-origin files are network-first and kept for offline use.
// The model server and the weather service are not cached; offline, the chat falls back to the DAE list on the phone.
const CACHE = "kg-v4";
self.addEventListener("install", e => { e.waitUntil(caches.open(CACHE).then(c => c.addAll(["./", "./index.html", "./dae.json", "./frg.json"]))); self.skipWaiting(); });
self.addEventListener("activate", e => { e.waitUntil(caches.keys().then(ks => Promise.all(ks.filter(k => k !== CACHE).map(k => caches.delete(k))))); self.clients.claim(); });
self.addEventListener("fetch", e => {
  const u = new URL(e.request.url);
  if (e.request.method !== "GET" || u.origin !== location.origin) return;
  e.respondWith(fetch(e.request)
    .then(r => { if (r.ok){ const copy = r.clone(); caches.open(CACHE).then(c => c.put(e.request, copy)); } return r; })
    .catch(() => caches.match(e.request, {ignoreSearch: true}).then(m => m || caches.match("./index.html"))));
});
