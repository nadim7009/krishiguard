"""KrishiGuard model server on a free Hugging Face Space.

Starts the official llama.cpp server (prebuilt CPU binary from its GitHub release) with Gemma 4 E2B in 4-bit,
and exposes its OpenAI-style /v1/chat/completions on port 7860 for the KrishiGuard website.
A small test chat sits at / so the Space page shows something useful. Messages are not stored.
"""
import io, os, stat, subprocess, threading, time, zipfile, tarfile, glob
import spaces  # free Gradio Spaces run on ZeroGPU hardware; the model itself runs on the CPU with llama.cpp
import httpx, gradio as gr
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.responses import StreamingResponse, JSONResponse

MODEL = "ggml-org/gemma-4-E2B-it-GGUF:Q4_0"
BIN_DIR = "/tmp/llama"
LLAMA = "http://127.0.0.1:8080"

def fetch_llama_server():
    """Download a recent llama.cpp Ubuntu x64 CPU build once (asset names change between releases, so search for one)."""
    found = glob.glob(f"{BIN_DIR}/**/llama-server", recursive=True)
    if found:
        return found[0]
    rels = httpx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15", timeout=60).json()
    skip = ("vulkan", "cuda", "rocm", "hip", "sycl", "arm", "aarch64", "openvino", "musa", "s390")
    asset = next(a for r in rels for a in r.get("assets", [])
                 if "ubuntu" in a["name"] and "x64" in a["name"] and not any(k in a["name"].lower() for k in skip)
                 and a["name"].endswith((".zip", ".tar.gz")))
    print("llama.cpp build:", asset["name"], flush=True)
    data = httpx.get(asset["browser_download_url"], timeout=600, follow_redirects=True).content
    if asset["name"].endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(BIN_DIR)
    else:
        tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(BIN_DIR)
    for f in glob.glob(f"{BIN_DIR}/**/*", recursive=True):
        if os.path.isfile(f):
            os.chmod(f, os.stat(f).st_mode | stat.S_IEXEC)
    return glob.glob(f"{BIN_DIR}/**/llama-server", recursive=True)[0]

LORA_URL = "https://raw.githubusercontent.com/nadim7009/krishiguard/main/models/krishiguard_lora_06_q8_0.gguf"
LORA = "/tmp/krishiguard_lora.gguf"

def fetch_lora():
    """KrishiGuard fine-tune (notebook 06, converted in notebook 09). Without it the Space still runs the base model."""
    try:
        if not os.path.exists(LORA):
            r = httpx.get(LORA_URL, timeout=300, follow_redirects=True)
            r.raise_for_status()
            open(LORA, "wb").write(r.content)
        print("LoRA:", os.path.getsize(LORA), "bytes", flush=True)
        return ["--lora", LORA]
    except Exception as e:
        print("LoRA not loaded, base model only:", e, flush=True)
        return []

def start_llama():
    exe = fetch_llama_server()
    env = dict(os.environ, LLAMA_CACHE="/tmp/llama-cache", LD_LIBRARY_PATH=os.path.dirname(exe))
    subprocess.Popen([exe, "-hf", MODEL, "--no-mmproj", *fetch_lora(), "--host", "127.0.0.1", "--port", "8080",
                      "-c", "4096", "-t", "2", "--parallel", "1", "--no-webui"], env=env)

threading.Thread(target=start_llama, daemon=True).start()

@spaces.GPU
def _gpu_probe():
    """ZeroGPU Spaces expect at least one GPU function; the chat itself does not use the GPU yet."""
    return "ok"

CORS = {}  # Gradio's middleware adds the CORS headers

async def health(request: Request):
    try:
        r = httpx.get(f"{LLAMA}/health", timeout=3)
        return JSONResponse({"model": MODEL, "lora": os.path.exists(LORA), "llama": r.json()}, status_code=r.status_code, headers=CORS)
    except Exception:
        return JSONResponse({"model": MODEL, "llama": "starting"}, status_code=503, headers=CORS)

async def chat(request: Request):
    """OpenAI-style chat, streamed straight through from llama.cpp."""
    body = await request.body()
    client = httpx.AsyncClient(timeout=None)
    req = client.build_request("POST", f"{LLAMA}/v1/chat/completions", content=body, headers={"content-type": "application/json"})
    try:
        r = await client.send(req, stream=True)
    except httpx.ConnectError:
        await client.aclose()
        return JSONResponse({"error": "model is still loading, try again in a minute"}, status_code=503, headers=CORS)
    async def relay():
        try:
            async for chunk in r.aiter_raw():
                yield chunk
        finally:
            await r.aclose(); await client.aclose()
    return StreamingResponse(relay(), status_code=r.status_code, headers=CORS, media_type=r.headers.get("content-type", "application/json"))

def test_chat(message, history):
    msgs = [{"role": "system", "content": "তুমি কৃষিগার্ড, বাংলাদেশের কৃষকদের বন্ধুর মতো সহকারী। সহজ বাংলায় ছোট উত্তর দেবে। বালাইনাশকের মাত্রা নিজে বানিয়ে বলবে না।"}]
    for m in history:  # Gradio passes dicts (messages) or [user, bot] pairs depending on version
        if isinstance(m, dict):
            msgs.append({"role": m["role"], "content": str(m["content"])})
        else:
            msgs += [{"role": "user", "content": str(m[0])}, {"role": "assistant", "content": str(m[1])}]
    msgs.append({"role": "user", "content": message})
    try:
        r = httpx.post(f"{LLAMA}/v1/chat/completions", json={"messages": msgs, "max_tokens": 400, "temperature": 0.6}, timeout=300)
        return r.json()["choices"][0]["message"]["content"]
    except Exception:
        return "মডেল এখনো চালু হচ্ছে। এক-দুই মিনিট পরে আবার চেষ্টা করুন।"

demo = gr.ChatInterface(test_chat, title="KrishiGuard model test",
                        description="Gemma 4 E2B (4-bit, llama.cpp). The full app is the KrishiGuard website; this page is for testing the model.")
demo.launch(prevent_thread_lock=True, ssr_mode=False)  # plain Python server, so our extra routes are reachable
# put our routes in front of Gradio's catch-all page route; Gradio's own CORS middleware answers the browser preflight
for path, fn, methods in [("/health", health, ["GET"]), ("/v1/chat/completions", chat, ["POST"])]:
    demo.app.router.routes.insert(0, APIRoute(path, fn, methods=methods))
demo.block_thread()
