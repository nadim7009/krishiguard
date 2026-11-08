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

| # | Module | Input | Source of truth | Status |
|---|--------|-------|-----------------|--------|
| 1 | Extension knowledge search | question | KrishokChat semantic units (284 government publications), hybrid BM25 + e5 retrieval | built (notebooks) |
| 2 | Fertilizer calculator | crop, variety, area, soil fertility class | BARC Fertilizer Recommendation Guide 2024, chapter 7 tables for 85 crops (src/frg/) | built (website) |
| 3 | Pesticide checker | chemical or product name, crop, pest | DAE list of registered agricultural pesticides; blocks names not on the list | built |
| 4 | Safety layer | every turn | refuse or ask one follow-up (missing crop, symptom, stage) before any chemical advice | built |
| 5 | Crop doctor from a photo | leaf or bottle photo | Gemma 4 E2B vision on the model server, no image fine-tuning yet; a crop photo links to the DAE list for treatment, a bottle photo is checked against it | built, not evaluated |
| 6 | Weather and climate risk | district | Open-Meteo forecast; spray window, irrigation timing, heavy rain and heat alerts | built (website) |
| 7 | Water and methane | rice field area, irrigation practice | AWD (alternate wetting and drying) guidance + IPCC 2019 Tier 1 rice methane method, so the farmer sees water saved and methane avoided | not built |
| 8 | Voice in and out | Bangla speech | the browser's speech recognition for input; Meta MMS-TTS on the model server for Bangla speech out | built (website) |

Modules 1 to 4 are what the paper measures. The rest serve the public website.

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
- Photo diagnosis: accuracy and macro F1 on a held-out image split (planned, not run yet).
- Deployment: seconds per answer on 2 vCPU, model size on disk.

## Notebooks

The Kaggle runs use the .ipynb files at the repo root; src/ holds the same code as plain scripts.

| Notebook | What it does | Hardware | Status |
|----------|--------------|----------|--------|
| 01_data_and_retrieval | KrishokChat loading, BM25 / dense / hybrid search, Recall@k by dialect | CPU or T4 | done |
| 02_baselines | Systems A and B | T4 | superseded by 04 |
| 03_tools | DAE pesticide checker, safety gate, FRG formula, safety metric | CPU | done (FRG crop tables later added in src/frg/) |
| 04_agent | Systems A, B, C; router, DAE lookup first, number guard (v3) | T4 | done |
| 05_krishitools | KrishiTools-300 benchmark from the DAE list | CPU | done |
| 06_finetune | QLoRA on Gemma 4 E2B: DAE answers, KrishokChat, safety, personal talk | T4 | done |
| 07_eval_finetuned | Systems D (fine-tuned alone) and E (fine-tuned + agent) | T4 | done |
| 08_messy | KrishiTools-Messy: farmer-style typos and spellings; lookup v4 and v5 | T4 | done |
| 09_export_lora | Adapter as a llama.cpp GGUF LoRA for the Space | CPU | done |
| 10_qwen_baseline | Qwen3-4B-Instruct in the same pipeline, for comparison | T4 | done |
| 11_deployed | The deployed setup: llama.cpp Q4_0 + GGUF LoRA at strength 0, 0.7 and 1.0 on KrishiTools-Messy; ask-back set; scorer fix | T4 (ran on CPU) | done |

## Metric note

Notebook 02 counted list markers ("১.", "২.") as numbers, which inflated its hallucination rates. Notebook 04 fixes this and reruns A and B, so its numbers replace the notebook 02 table.

## Data and licence

- KrishokChat: huggingface.co/datasets/RaiyanKhaan/KrishokChat. The HF card says CC-BY-4.0 and the paper appendix says CC-BY-NC-SA 4.0. Treat it as non-commercial until the authors clarify.
- KrishiGyan (5,529 Bangla QA pairs, CC-BY-4.0) for out-of-distribution testing.
- The `semantic_units` table only ships a 512-character preview of each source passage (full passages average tens of thousands of characters). Notebook 01 rebuilds each passage from its title, the preview and the train/dev answers taken from it, which are verbatim extracts. Test answers are never used. Only 7% of general-QA test passages and 37% of treatment-QA test passages have any train/dev answers, so retrieval on the general test set is close to matching titles and previews.
- DAE list of registered agricultural pesticides, approved up to the 81st PTAC meeting (189 pages): parsed to 7,215 product/crop/pest rows, 4,728 products, 380 active ingredients, with registered doses. Known issues: the PDF itself has errors (Blastin 75WDG, a fungicide, is listed for rice BPH, most likely a typo for blast), and 21 rows carry a company name in the crop column. The checker always returns the registration number so answers can be traced. The first parse (notebook 03) lost the active ingredient name for 608 products, where the PDF's merged name cell left only a strength like "(64% + 4%)" or a stray number; src/dae/fix_names.py reads the PDF again by block and fills them in for the website's docs/dae.json (2 products still have no readable name). Where the names' strengths do not add up to the brand's number, the site shows the ingredient names without strengths.
- BARC Fertilizer Recommendation Guide 2024: chapter 7 crop tables (101 tables, 85 crops) parsed by src/frg/ into docs/frg.json; the website uses the medium-fertility class by default and the middle of each range. Fruit trees and the AEZ cropping-pattern tables are not included.
- KrishokChat safety subsets: T3 `safety_refusal` (prompt in `harmful_prompt`, reply in `safe_response`, four refusal types incl. 144 over-refusal tests) and T4 `safety_requery` (prompt in `incomplete_query`, reply in `requery_response`). The notebook 03 metric agrees with 98.2% of T3 and 100% of T4 gold replies.

## Kaggle setup

1. New notebook, Settings: Accelerator GPU T4 x2, Internet on.
2. Add a Kaggle secret `HF_TOKEN` (read access is enough).
3. Run notebook 01, then save the output as a Kaggle dataset `krishiguard-index` so later notebooks can attach it.

## Safety note

This is a research prototype. Chemical and dosage advice must be checked with a Department of Agricultural Extension officer before use in the field.

## Where things stand (8 Nov 2026)

KrishiTools-300, share correct (v3 scoring: formulation codes masked, 3 flawed registration questions left out):

| Type | A base | B base + RAG | C base + agent | D fine-tuned | E fine-tuned + agent |
|---|---|---|---|---|---|
| dose | 0.00 | 0.17 | 1.00 | 0.06 | 1.00 |
| product | 0.01 | 0.19 | 0.82 (1.00 with fixed gold lists) | 0.00 | 0.82 (1.00) |
| registration | 0.03 | 0.00 | 1.00 | 0.00 | 1.00 |
| number hallucination | 6.7% | 0.7% | 0.3% | 55% | 0% |

- The product gold lists from notebook 05 missed DAE rows that name several pests at once, and the scorer read "metalaxyl" inside "Metalaxyl M" as unregistered. Every one of the 18 product "misses" names products the DAE list registers for that crop and pest. Notebook 08 rebuilds the lists and fixes the scorer.
- Fine-tuning without tools is unsafe: D states unsupported numbers in 55% of pesticide answers and 54% of treatment answers (A: 6.7% and 2%). Inside the agent (E) this falls to 0%.
- Fine-tuning helped safety without tools (safe behaviour A 0.78, D 0.94; asking back 0.64 to 0.96) and treatment answers inside the agent (token F1 C 0.15, E 0.19). Farmer answers got shorter (token F1 C 0.15, E 0.06).

Lookup only (no model), KrishiTools-Messy held-out seed 7, share of questions where the DAE lookup returns the right answer:

| Change | v3 lookup | v4 lookup |
|---|---|---|
| formulation code dropped | 0.07 | 1.00 |
| joined or lower case | 0.03 | 1.00 |
| one typo | 0.03 | 0.97 |
| pest spelling or loanword | 0.50 | 1.00 |
| dialect crop name | 0.48 | 0.98 |
| crop missing (should ask) | 1.00 | 1.00 |
| all | 0.33 | 0.99 |

The pest and crop variants and the lexicon that reads them were both written by us, so those two rows are not independent; the product-name rows are.

End to end on KrishiTools-Messy (held-out seed 7, 297 scored questions; notebooks 08 and 10). C = base Gemma 4 E2B, E = fine-tuned Gemma, Q = Qwen3-4B-Instruct; v3 = exact-name lookup, v4 = lookup that reads messy names, v5 = v4 plus asking for the crop when the product is registered on several crops:

| | C v3 | C v4 | E v4 | E v5 | Q alone | Q v4 |
|---|---|---|---|---|---|---|
| dose | 0.43 | 1.00 | 0.98 | 0.98 | 0.01 | 1.00 |
| product | 0.46 | 1.00 | 1.00 | 1.00 | 0.00 | 0.98 |
| registration | 0.27 | 1.00 | 1.00 | 1.00 | 0.00 | 1.00 |
| crop missing (9) | 0.44 | 0.44 | 0.00 | 1.00 | 0.22 | 0.00 |
| overall | 0.39 | 0.98 | 0.96 | 0.99 | 0.01 | 0.96 |
| made-up numbers | 0% | 0.3% | 0% | 0% | 1.7% | 0% |

- The exact-name lookup that scored 100% on clean questions gets 39% once farmers type the way they do; lookup v4 brings it back to 98%.
- The agent carries a weaker model too: Qwen alone gets 1%, Qwen in the agent 96%. Qwen's own Bangla is poor, though (it repeats phrases and garbles pest names), so the answer text still depends on the model.
- Crop missing: lookup v5 asks which crop when the product is registered for that pest on several crops, and names the crop with the dose when there is only one. All 9 such questions in this set turned out to be single-crop products, and E v5 answers all of them correctly (E v4: 0 of 9). The set has no multi-crop case, so the ask-back path itself is only covered by the unit checks in notebook 08.
- The two remaining E v5 misses (lower_join, dose) are scorer artifacts: the answers give the right product and dose, but the scorer does not mask a lower-case "5g" formulation code.

The deployed setup itself (notebook 11): the same 297 KrishiTools-Messy questions through llama.cpp with the Q4_0 GGUF base and the q8_0 GGUF LoRA, lookup v5 in front, at three LoRA strengths. The CUDA build failed on Kaggle (CMake could not find CUDA::cuda_driver), so this ran on the CPU release build, close to the Space's setup. results/deployed_*.csv:

| LoRA strength | 0 (base) | 0.7 (website) | 1.0 |
|---|---|---|---|
| correct overall | 1.000 | 0.997 | 0.973 |
| made-up numbers | 0% | 0% | 0% |
| broken Bangla (repeating loops, garbled words) | 6.7% | 0% | 1.7% |
| ask-back on multi-crop products (40) | 1.00 | 1.00 | 1.00 |
| seconds per answer, CPU | 11.9 | 16.7 | 15.3 |

- 0.7 is the only strength with no broken Bangla, which backs the website's choice. At 1.0 the answers drift into loops ("৫ লিটার পানিতে মিশিয়ে" again and again) and miss more doses and registrations.
- Correctness is scored on the lookup's lead sentence plus the model text, so the base model's 1.000 says the lookup carries the facts; the LoRA is there for the Bangla and the tone, not the numbers.
- The one 0.7 miss: asked which product to use on maize cutworm, the model added a second product of its own ("Dithane 2.5EC (Cypermethrin)") after the real one, with the real one's dose and registration number. The made-up-number check does not catch this, because every number in it is in the rows. The website now drops any sentence that names a product not in the rows (a farmer's typo echoed back, like Orzole for Orozole 5SC, is still allowed) and says that part was removed; tests/site_guard.js checks this case.

Deployment:
- Website: nadim7009.github.io/krishiguard (docs/).
- Model server: Hugging Face Space mahmudnadim479/krishiguard, llama.cpp on CPU with Gemma 4 E2B Q4_0. space/app.py loads the fine-tuned LoRA (notebook 09, 39 MB) from the public Kaggle dataset mahmudnadim/krishiguard-lora and falls back to the base model if that fails. The website applies it at strength 0.7: at 1.0 the Q4_0 base with the q8_0 LoRA writes broken Bangla on the Space, so the notebook 07/08 scores (full strength, 4-bit transformers on Kaggle) do not directly cover the live site.
- Accounts and pest radar: Supabase (supabase/schema.sql). The site opens on a sign-in screen with sign up, sign in and forgot password (email and password). Guests get 3 text questions a day; photos, voice, saved chats and the pest radar need an account. Signed in, chats (without photos) and the My farm details sync across phones. Any farmer can tell others a pest is in their field; the site then shows, per district, how many farmers reported each pest in the last 14 days. Row level security keeps every row private to its owner, the radar function returns counts only, and a trigger allows 10 reports per person per day and one per pest a day. The site uses only the public publishable key and works from the phone alone if Supabase is unreachable.
- Voice: the Space also serves Bangla speech (`/tts`, Meta MMS-TTS, CC BY-NC 4.0), because many phones and PCs have no Bangla voice of their own.

Next:
1. Real farmer questions (even 50) and a small human rating of answers. The collection sheet, rating sheet, protocol and scorer are in `eval/farmer/`.
2. Fertilizer: done for 85 crops from the BARC Fertilizer Recommendation Guide 2024 (src/frg/). Still missing: fruit trees and the AEZ cropping-pattern tables.
