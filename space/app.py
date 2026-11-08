"""KrishiGuard model server on a free Hugging Face Space.

Starts the official llama.cpp server (prebuilt CPU binary from its GitHub release) with Gemma 4 E2B in 4-bit,
with its vision projector (leaf and bottle photos) and the KrishiGuard LoRA,
and exposes its OpenAI-style /v1/chat/completions on port 7860 for the KrishiGuard website.
A small test chat sits at / so the Space page shows something useful. Messages are not stored.
"""
import io, os, stat, subprocess, threading, time, zipfile, tarfile, glob, asyncio, wave
import spaces  # free Gradio Spaces run on ZeroGPU hardware; the model itself runs on the CPU with llama.cpp
import httpx, gradio as gr
from fastapi import Request
from fastapi.routing import APIRoute
from fastapi.responses import StreamingResponse, JSONResponse, Response
from bn_speech import speakable

MODEL = "ggml-org/gemma-4-E2B-it-GGUF:Q4_0"
BIN_DIR = "/tmp/llama"
LLAMA = "http://127.0.0.1:8080"

def fetch_llama_server():
    """Download a recent llama.cpp Ubuntu x64 CPU build once (asset names change between releases, so search for one)."""
    found = glob.glob(f"{BIN_DIR}/**/llama-server", recursive=True)
    if found:
        return found[0]
    skip = ("vulkan", "cuda", "rocm", "hip", "sycl", "arm", "aarch64", "openvino", "musa", "s390")
    try:
        rels = httpx.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15", timeout=60,
                         follow_redirects=True, headers={"User-Agent": "krishiguard-space"}).json()
        asset = next(a for r in rels for a in r.get("assets", [])
                     if "ubuntu" in a["name"] and "x64" in a["name"] and not any(k in a["name"].lower() for k in skip)
                     and a["name"].endswith((".zip", ".tar.gz")))
        name, url = asset["name"], asset["browser_download_url"]
    except Exception as e:  # GitHub API rate limit or outage: use a build known to work (also used in notebook 09)
        print("release lookup failed, using pinned build:", e, flush=True)
        name = "llama-b12189-bin-ubuntu-x64.tar.gz"
        url = f"https://github.com/ggml-org/llama.cpp/releases/download/b12189/{name}"
    print("llama.cpp build:", name, flush=True)
    data = httpx.get(url, timeout=600, follow_redirects=True).content
    if name.endswith(".zip"):
        zipfile.ZipFile(io.BytesIO(data)).extractall(BIN_DIR)
    else:
        tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(BIN_DIR)
    for f in glob.glob(f"{BIN_DIR}/**/*", recursive=True):
        if os.path.isfile(f):
            os.chmod(f, os.stat(f).st_mode | stat.S_IEXEC)
    return glob.glob(f"{BIN_DIR}/**/llama-server", recursive=True)[0]

LORA_DATASET, LORA_FILE = "mahmudnadim/krishiguard-lora", "krishiguard/krishiguard_lora_06_q8_0.gguf"  # public Kaggle dataset (notebook 09)
LORA = "/tmp/krishiguard_lora.gguf"

def fetch_lora():
    """KrishiGuard fine-tune (notebook 06, converted in notebook 09). Without it the Space still runs the base model."""
    try:
        if not os.path.exists(LORA):
            import kagglehub, shutil  # public datasets download without a Kaggle account
            shutil.copy(kagglehub.dataset_download(LORA_DATASET, path=LORA_FILE), LORA)
        print("LoRA:", os.path.getsize(LORA), "bytes", flush=True)
        return ["--lora", LORA]
    except Exception as e:
        print("LoRA not loaded, base model only:", e, flush=True)
        return []

def start_llama():
    exe = fetch_llama_server()
    env = dict(os.environ, LLAMA_CACHE="/tmp/llama-cache", LD_LIBRARY_PATH=os.path.dirname(exe))
    subprocess.Popen([exe, "-hf", MODEL, *fetch_lora(),  # -hf also fetches the vision projector, for leaf and bottle photos
                      "--host", "127.0.0.1", "--port", "8080",
                      "-c", "4096", "-t", "2", "--parallel", "1", "--no-webui"], env=env)

threading.Thread(target=start_llama, daemon=True).start()

# Bangla voice for the website's voice mode. Phones and PCs often have no Bangla voice of their own, and then the
# browser reads Bangla with an English voice, which sounds like no language at all. MMS-TTS (Meta, CC BY-NC 4.0) is
# small enough for the CPU: about a second per short sentence.
TTS = {}
def load_tts():
    if "model" not in TTS:
        import torch
        from transformers import VitsModel, AutoTokenizer
        torch.set_num_threads(2)
        TTS["tok"] = AutoTokenizer.from_pretrained("facebook/mms-tts-ben")
        TTS["model"] = VitsModel.from_pretrained("facebook/mms-tts-ben").eval()
        print("TTS ready", flush=True)
    return TTS["tok"], TTS["model"]
threading.Thread(target=load_tts, daemon=True).start()
TTS_LOCK = threading.Lock()

def synth(text):
    import torch, numpy as np
    tok, model = load_tts()
    rate, pause, parts = model.config.sampling_rate, None, []
    pause = np.zeros(int(rate * 0.28), dtype=np.float32)
    with TTS_LOCK, torch.no_grad():
        for phrase in speakable(text)[:12]:
            ids = tok(phrase, return_tensors="pt")
            if ids["input_ids"].shape[1] < 2:
                continue
            parts += [model(**ids).waveform[0].numpy(), pause]
    if not parts:
        return None
    pcm = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate); w.writeframes(pcm)
    return buf.getvalue()

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

async def tts(request: Request):
    """Bangla text in, WAV out."""
    try:
        text = str((await request.json()).get("text", ""))[:700]
    except Exception:
        return JSONResponse({"error": "send JSON {\"text\": ...}"}, status_code=400, headers=CORS)
    try:
        audio = await asyncio.to_thread(synth, text)
    except Exception as e:
        print("tts failed:", e, flush=True)
        return JSONResponse({"error": "voice is not ready yet"}, status_code=503, headers=CORS)
    if not audio:
        return JSONResponse({"error": "nothing to say"}, status_code=400, headers=CORS)
    return Response(audio, media_type="audio/wav", headers=CORS)

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
for path, fn, methods in [("/health", health, ["GET"]), ("/v1/chat/completions", chat, ["POST"]), ("/tts", tts, ["POST"])]:
    demo.app.router.routes.insert(0, APIRoute(path, fn, methods=methods))
demo.block_thread()
