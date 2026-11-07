# %% [markdown]
# # KrishiGuard 06: fine-tune Gemma 4 E2B (QLoRA) for systems D and E
#
# The agent (system C) gets the right DAE rows but the base model still copies doses from other passages, asks back when it does not need to, or ignores the amount already worked out for the farmer.
# This notebook teaches the model three habits: answer from the DAE rows it is given and nothing else, say a clear no without any dose when a product is not registered, and talk with a farmer about anything (money, family, a bad season) kindly and without "ভাই" unless the farmer uses it.
#
# Training data, all built here:
# - DAE answers in two prompt formats: the agent's (notebook 04) and the website's. Products used in KrishiTools-300 dose and registration questions are held out, and the question templates differ from the benchmark's.
# - KrishokChat general and treatment QA, train and dev splits only, half of them with the gold passage among two distractors.
# - KrishokChat safety refusal and re-query rows, minus the ones notebook 04 samples for evaluation.
# - A small hand-written set of personal conversations, including a self-harm reply that points to 999.
#
# Output: the LoRA adapter and the training data. D (adapter alone) and E (adapter inside the agent) are scored in notebook 04.
#
# Kaggle settings: GPU T4 x2, Internet on, inputs: KrishiGuard 01 output (chunks), 03 output (DAE csv), 05 output (krishitools_300.jsonl).

# %%
!pip install -q -U "bitsandbytes>=0.46.1" peft  # 4-bit loading; must run before transformers is first imported
import os, re, json, time, glob, random
import numpy as np, pandas as pd, torch
from datasets import load_dataset

REPO = "RaiyanKhaan/KrishokChat"
OUT = "/kaggle/working/krishiguard"
os.makedirs(OUT, exist_ok=True)
SEED, N_EVAL = 42, 50   # N_EVAL and SEED must match notebook 04 so its evaluation rows stay out of training
rng = random.Random(SEED)
find = lambda name: glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0]
