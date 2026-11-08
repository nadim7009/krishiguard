---
title: KrishiGuard
emoji: 🌾
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.31.0
python_version: '3.12'
app_file: app.py
pinned: false
short_description: Bangla farm assistant model server (Gemma 4 E2B)
---

# KrishiGuard model server

Runs Gemma 4 E2B (4-bit GGUF from ggml-org) with the official llama.cpp server on the CPU. The KrishiGuard website sends each farmer message here together with the farmer's saved profile, the matching rows from the DAE list of registered pesticides and the local weather, and streams the Bangla answer back.

- `POST /v1/chat/completions`: OpenAI format, `stream: true` supported, open to the KrishiGuard website (CORS)
- `GET /health`: whether the model has finished loading
- `POST /tts`: Bangla text in, WAV audio out (Meta MMS-TTS `facebook/mms-tts-ben`, CC BY-NC 4.0), for the website's voice mode
- The page at `/` is a small test chat.

Doses and product names never come from the model alone: the website only shows amounts that appear in the DAE rows it sent. Conversations are not stored by this Space.

Research prototype. Check chemical and dose advice with a Department of Agricultural Extension officer before use.
