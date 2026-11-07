# %% [markdown]
# # KrishiGuard 09: the fine-tuned adapter as a llama.cpp LoRA for the website
#
# The Hugging Face Space runs Gemma 4 E2B Q4_0 with llama.cpp on a CPU. Instead of merging and re-quantising the whole model, the notebook 06 adapter is converted to a small GGUF LoRA (about 25 MB)
# that llama.cpp applies on top of the same base file at load time.
# The last cell starts llama-server with the base model and the LoRA and asks the same website-style question with the LoRA on and off, as a check that the conversion kept what training learned.
#
# Kaggle settings: no accelerator needed, Internet on, input: KrishiGuard 06 (adapter_06).

# %%
import os, glob, json, time, subprocess, io, zipfile, tarfile, stat
import requests

OUT = "/kaggle/working/krishiguard"
os.makedirs(OUT, exist_ok=True)
ADAPTER = os.path.dirname(glob.glob("/kaggle/input/**/adapter_06/adapter_config.json", recursive=True)[0])
print("adapter:", ADAPTER, json.load(open(f"{ADAPTER}/adapter_config.json"))["base_model_name_or_path"])
subprocess.run("git clone -q --depth 1 https://github.com/ggml-org/llama.cpp /kaggle/working/llama.cpp && "
               "pip install -q /kaggle/working/llama.cpp/gguf-py", shell=True, check=True)

# %%
LORA = f"{OUT}/krishiguard_lora_06_q8_0.gguf"
r = subprocess.run(["python", "/kaggle/working/llama.cpp/convert_lora_to_gguf.py", ADAPTER, "--outfile", LORA, "--outtype", "q8_0"],
                   capture_output=True, text=True)
print(r.stdout[-3000:], r.stderr[-3000:])
assert r.returncode == 0 and os.path.exists(LORA), "conversion failed"
print(round(os.path.getsize(LORA) / 1e6, 1), "MB")

# %% [markdown]
# ## Check: same question, LoRA on and off

# %%
def fetch_llama_server(bin_dir="/kaggle/working/llama-bin"):
    """Recent llama.cpp Ubuntu x64 CPU build, the same way the Space gets it."""
    rels = requests.get("https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15", timeout=60).json()
    skip = ("vulkan", "cuda", "rocm", "hip", "sycl", "arm", "aarch64", "openvino", "musa", "s390")
    asset = next(a for rel in rels for a in rel.get("assets", [])
                 if "ubuntu" in a["name"] and "x64" in a["name"] and not any(k in a["name"].lower() for k in skip)
                 and a["name"].endswith((".zip", ".tar.gz")))
    data = requests.get(asset["browser_download_url"], timeout=600).content
    (zipfile.ZipFile(io.BytesIO(data)).extractall(bin_dir) if asset["name"].endswith(".zip")
     else tarfile.open(fileobj=io.BytesIO(data), mode="r:gz").extractall(bin_dir))
    for f in glob.glob(f"{bin_dir}/**/*", recursive=True):
        if os.path.isfile(f):
            os.chmod(f, os.stat(f).st_mode | stat.S_IEXEC)
    print("llama.cpp build:", asset["name"])
    return glob.glob(f"{bin_dir}/**/llama-server", recursive=True)[0]

exe = fetch_llama_server()
env = dict(os.environ, LLAMA_CACHE="/kaggle/working/llama-cache", LD_LIBRARY_PATH=os.path.dirname(exe))
server = subprocess.Popen([exe, "-hf", "ggml-org/gemma-4-E2B-it-GGUF:Q4_0", "--no-mmproj", "--lora", LORA,
                           "--port", "8080", "-c", "4096", "-t", "4", "--no-webui"], env=env,
                          stdout=open(f"{OUT}/llama_server.log", "w"), stderr=subprocess.STDOUT)
for _ in range(120):
    try:
        if requests.get("http://127.0.0.1:8080/health", timeout=3).ok:
            break
    except requests.RequestException:
        pass
    time.sleep(10)
print(requests.get("http://127.0.0.1:8080/lora-adapters", timeout=10).json())

SYSTEM = (
    "তুমি কৃষিগার্ড, বাংলাদেশের কৃষকদের পাশে থাকা একজন বন্ধুর মতো সহকারী। বালাইনাশকের নাম, মাত্রা বা কোনো পরিমাণ কখনো নিজে বানিয়ে বলবে না। "
    "শুধু নিচে দেওয়া DAE তথ্য থেকে বলবে। কৃষক নিজে না বললে \"ভাই\" সম্বোধন করবে না।\n\n"
    "এই কৃষক নিজে জানিয়েছেন: নাম করিম; ফসল ধান; জমি ৩ বিঘা।\n\n"
    "DAE নিবন্ধিত বালাইনাশক তালিকা থেকে এই প্রশ্নের তথ্য (ওষুধের নাম ও মাত্রা শুধু এখান থেকে নেবে):\n"
    "পণ্য Taba 25WDG (Ethion (5%) + Thiamethoxam (20%) · Rice, BPH): নিবন্ধিত মাত্রা 100 g/ha; এই কৃষকের জন্য হিসাব করা: আপনার ৩ বিঘা জমিতে মোট প্রায় ৪০ গ্রাম\n"
    "নিবন্ধন নম্বর: DAE AP-4578")
checks = {}
for name, scale in [("lora_on", 1.0), ("lora_off", 0.0)]:
    body = {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "ধানে কারেন্ট পোকা লেগেছে, Taba 25WDG কতটুকু দিব?"}],
            "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}, "lora": [{"id": 0, "scale": scale}]}
    checks[name] = requests.post("http://127.0.0.1:8080/v1/chat/completions", json=body, timeout=300).json()["choices"][0]["message"]["content"]
    print(f"[{name}]", checks[name], "\n")
json.dump(checks, open(f"{OUT}/lora_check.json", "w"), ensure_ascii=False, indent=1)
assert "৪০ গ্রাম" in checks["lora_on"] or "40" in checks["lora_on"], "the LoRA answer did not use the amount worked out for the farmer"
server.terminate()
