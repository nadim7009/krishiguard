# KrishiGuard

An offline Bangla agent for farmers that handles crops, water, weather and emissions in one place. The key design rule: the language model never makes up a number. Every dose, rate, date or emission figure comes from a tool that reads an official source or runs a fixed formula, and the answer shows which tool it came from.

Built on Kaggle (free T4 GPUs) with Gemma 4 E2B, deployed on Hugging Face Spaces (free CPU tier).

## Why this is different from a farm chatbot

Most agri LLM projects are a chat box over documents. The KrishokChat benchmark (Reza, Nimi and Shahid, arXiv 2606.29243) shows where that breaks for Bangla:

1. No retriever was tested end to end; models were either closed-book or given the correct passage.
2. Even with the correct passage, every model invented a chemical or dose in 4 to 7 percent of treatment answers.
3. After answer-only fine-tuning, their 4B model refused or asked a follow-up on 1 of 323 unsafe questions.

KrishiGuard treats point 2 as a design problem rather than a training problem: numbers are moved out of the model and into tools.

## Research question

Does a small on-device agent that routes every numeric claim through verified tools cut dosage and number hallucination to near zero, compared with retrieval alone, without losing answer quality or refusal behaviour?

## Modules

Each module is a tool the agent can call. The model decides which tool to use, fills in the arguments, and writes the Bangla answer around the tool output.

| # | Module | Input | Source of truth | Phase |
|---|--------|-------|-----------------|-------|
| 1 | Extension knowledge search | question | KrishokChat semantic units (284 government publications), hybrid BM25 + e5 retrieval | 1 |
| 2 | Fertilizer calculator | crop, variety, area, soil fertility class | Fertilizer Recommendation Guide 2018 (BARC) tables, coded as rules | 1 |
| 3 | Pesticide checker | chemical or product name, crop, pest | DAE list of registered agricultural pesticides; blocks names not on the list | 1 |
| 4 | Safety layer | every turn | refuse or ask one follow-up (missing crop, symptom, stage) before any chemical advice | 1 |
| 5 | Crop doctor from a photo | leaf photo | Gemma 4 E2B vision, QLoRA on Bangladesh crop disease images; links the label to module 1 for treatment | 2 |
| 6 | Weather and climate risk | district or upazila | Open-Meteo forecast; spray window, irrigation timing, heavy rain and heat alerts | 2 |
| 7 | Water and methane | rice field area, irrigation practice | AWD (alternate wetting and drying) guidance + IPCC 2019 Tier 1 rice methane method, so the farmer sees water saved and methane avoided | 2 |
| 8 | Voice in and out | Bangla speech | open Bangla speech-to-text and text-to-speech models, for farmers who do not type | 3 |

Phase 1 is the paper. Phase 2 makes the system clearly broader than a QA bot. Phase 3 is for the public demo.

## Experiment matrix

| ID | System | Retrieval | Tools | Fine-tune | Safety data |
|----|--------|-----------|-------|-----------|-------------|
| A | Base E2B, closed-book | no | no | no | no |
| B | Base E2B + RAG | yes | no | no | no |
| C | Base E2B agent | yes | yes | no | no |
| D | QLoRA, answers only | yes | yes | yes | no |
| E | QLoRA, answers + safety + tool-call traces | yes | yes | yes | yes |

The comparison that matters most is B against C and E on number hallucination.

## Evaluation

Reused from KrishokChat, so numbers line up with a published table:

- General QA: Bangla token F1, number hallucination
- Treatment QA: chemical precision / recall / F1, dosage hallucination
- Safety QA: refusal (T3) and follow-up question (T4) compliance
- Farmer benchmark: 350 real farmer questions

New benchmark built for this project (released with the paper):

- KrishiTools-300: about 300 Bangla farmer requests that need a tool, written across the six dialects in KrishokChat, each with the correct tool and arguments. Scored on tool choice accuracy, argument accuracy and final number correctness.
- Photo diagnosis: accuracy and macro F1 on a held-out image split (phase 2).
- Deployment: seconds per answer on 2 vCPU, model size on disk.

## Notebooks

| Notebook | What it does | Hardware | Status |
|----------|--------------|----------|--------|
| 01_data_and_retrieval | KrishokChat loading, BM25 / dense / hybrid search, Recall@k by dialect | CPU or T4 | ready |
| 02_baselines | Systems A and B (v6: general, treatment, farmer; v7: safety with the real prompts) | T4 | done |
| 03_tools | DAE pesticide checker, safety gate, FRG formula, safety metric, each with tests | CPU | done (FRG tables pending) |
| 04_agent | System C: router (ask / refuse / answer), DAE tool evidence, number guard; reruns A and B with the fixed number metric | T4 | running |
| 05_finetune | Systems D and E, QLoRA with safety data and tool traces | T4 | |
| 06_krishitools_benchmark | Build and check the 300 tool-use scenarios | CPU | |
| 07_crop_doctor | Vision fine-tune and photo-to-treatment flow | T4 | phase 2 |
| 08_weather_and_methane | Open-Meteo and IPCC tools | CPU | phase 2 |
| 09_export_and_deploy | GGUF Q4_K_M, Gradio Space with voice | CPU | phase 3 |

## Metric note

Notebook 02 counted list markers ("১.", "২.") as numbers, which inflated its hallucination rates. Notebook 04 fixes this and reruns A and B, so its numbers replace the notebook 02 table.

## Data and licence

- KrishokChat: huggingface.co/datasets/RaiyanKhaan/KrishokChat. The HF card says CC-BY-4.0 and the paper appendix says CC-BY-NC-SA 4.0. Treat it as non-commercial until the authors clarify.
- KrishiGyan (5,529 Bangla QA pairs, CC-BY-4.0) for out-of-distribution testing.
- The `semantic_units` table only ships a 512-character preview of each source passage (full passages average tens of thousands of characters). Notebook 01 rebuilds each passage from its title, the preview and the train/dev answers taken from it, which are verbatim extracts. Test answers are never used. Only 7% of general-QA test passages and 37% of treatment-QA test passages have any train/dev answers, so retrieval on the general test set is close to matching titles and previews.
- DAE list of registered agricultural pesticides, approved up to the 81st PTAC meeting (189 pages): parsed to 7,215 product/crop/pest rows, 4,728 products, 380 active ingredients, with registered doses. Known issues: the PDF itself has errors (Blastin 75WDG, a fungicide, is listed for rice BPH, most likely a typo for blast), and 21 rows carry a company name in the crop column. The checker always returns the registration number so answers can be traced.
- BARC Fertilizer Recommendation Guide 2024: the formula is implemented; the crop tables still need the guide PDF.
- KrishokChat safety subsets: T3 `safety_refusal` (prompt in `harmful_prompt`, reply in `safe_response`, four refusal types incl. 144 over-refusal tests) and T4 `safety_requery` (prompt in `incomplete_query`, reply in `requery_response`). The notebook 03 metric agrees with 98.2% of T3 and 100% of T4 gold replies.

## Kaggle setup

1. New notebook, Settings: Accelerator GPU T4 x2, Internet on.
2. Add a Kaggle secret `HF_TOKEN` (read access is enough).
3. Run notebook 01, then save the output as a Kaggle dataset `krishiguard-index` so later notebooks can attach it.

## Safety note

This is a research prototype. Chemical and dosage advice must be checked with a Department of Agricultural Extension officer before use in the field.

## Where things stand (7 Nov 2026)

Done on Kaggle (all committed, nothing running):
- 01 retrieval: fine-tuned bge-m3 is the retriever.
- 02 baselines: superseded by 04 because of the list-marker metric bug.
- 03 tools: DAE pesticide parser and checker, safety gate, FRG formula, safety metric.
- 04 agent v1 (5 Nov): A, B, C on 50 questions per set.
- 05 KrishiTools-300: 100 dose, 100 product and 100 registration questions built from the DAE list.
- 04 agent v2 (Kaggle version 3, 7 Nov, 4 h 12 min on T4 x2): product-name lookup, router retrained with real farmer questions, scored on KrishiTools-300. Results in results/agent_v2_*.csv (copied from the run log; the per-question file is still on Kaggle).

KrishiTools-300, share correct:

| Type | A closed book | B RAG | C agent |
|---|---|---|---|
| dose | 0.00 | 0.16 | 0.50 |
| product | 0.01 | 0.19 | 0.36 |
| registration | 0.03 | 0.00 | 0.75 |

Number hallucination on KrishiTools: A 6.7%, B 0.7%, C 0%. C's tool fired on 100% of these questions; the number guard dropped a sentence in 0.7% of answers.
Farmer set: the router now asks back on 5/50 questions (14/50 in v1); C token F1 0.151 (v1 0.124, B 0.160).

Deployment started:
- Website (docs/, served by GitHub Pages): Bangla-first chat, pesticide checker over the parsed DAE list, leaf photo page (model pending), weather and spray windows, My farm profile kept on the phone, personal conversation, number guard on model text.
- Model server: Hugging Face Space mahmudnadim479/krishiguard (Gradio on ZeroGPU hardware, llama.cpp on CPU, Gemma 4 E2B Q4_0, thinking off). About 15 tokens per second; first token in about 1.3 s. Code in space/.

Next steps, in order:
1. Look at C's wrong dose and product answers on KrishiTools-300 (per-question file) and fix the tool or the prompt.
2. Publish the website on GitHub Pages (needs a repo name).
3. Notebook 05 fine-tune (systems D and E): safety data, tool traces, everyday and personal Bangla conversation, profile in the prompt, no "ভাই" unless the farmer uses it.
4. FRG 2024 tables (needs the BARC guide PDF).
5. Bottle check, rotation advice, pest radar, then voice and offline.
