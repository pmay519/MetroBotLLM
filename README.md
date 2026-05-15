# MetroBot 🤖📐

**A privately fine-tuned local LLM for the MetroStack industrial metrology backend.**

MetroBot is a sub-3B parameter code model, quantized to GGUF Q4_K_M and deployed via Ollama, that has been trained exclusively on MetroStack source code and synthetic instruction pairs. It understands your exact stack, refuses dangerous operations, and always explains its reasoning before writing a single line of code.

> "We don't need a God-model. We need a specialized, fast model that understands SQL, Docker, and the logic of the stack."

---

## What MetroBot Is

MetroBot is not a general-purpose assistant. It is a **domain-specific engineering tool** trained to understand and produce code for one project: MetroStack — a 3D scan processing and metrology analysis pipeline built on FastAPI, PostgreSQL + PostGIS + pgpointcloud, React, three.js, Open3D, and Docker.

It runs **fully offline** on a Snapdragon 8 device or an HP notebook with 16GB RAM. No API calls. No cloud dependency. No data leaves your machine.

---

## Stack MetroBot Knows

| Layer | Technology |
|---|---|
| Backend | FastAPI 0.111 / Python 3.12 / Uvicorn |
| Database | PostgreSQL 16 + PostGIS 3.4 + pgpointcloud 1.2 |
| DB Driver | asyncpg 0.29 (async only — psycopg2 is explicitly forbidden) |
| ORM / Migrations | SQLAlchemy 2.x async + Alembic |
| Frontend | React 18 + TypeScript 5 + Capacitor 6 (Android) |
| 3D Viewer | three.js r165 + react-three-fiber v8 + drei v9 |
| State | Zustand 4.x |
| Containers | Docker Compose V2 (`compose.yml`) |
| Point Cloud I/O | Open3D 0.18 (E57/PLY), trimesh 4.x |
| Numerics | numpy 1.26, scipy 1.13 |
| Auth | JWT via python-jose, secrets via environment variables |

**Forbidden (MetroBot will refuse or warn):** psycopg2, Redux, Flask, SQLite, MySQL, moment.js, Django

---

## Project Structure

```
metrobot/
├── metrostack_dataset_builder.py    # Phase 1: Harvest project files + teacher enrichment
├── metrostack_seeds.jsonl           # Hand-crafted seed training pairs (verified examples)
├── metrostack_synthetic_gen_v2.py  # Phase 2: 5-engine synthetic data generator (500+ pairs)
├── metrobot_train.py               # Phase 3: Unsloth QLoRA fine-tuning + GGUF export
├── Modelfile                        # Phase 4: Ollama deployment configuration
├── metrobot_smoke_test.py          # Phase 5: 20-test scoring suite with auto-diagnosis
└── README.md                        # This file
```

---

## How It Works

### The Training Pipeline

MetroBot is built in five phases:

```
Your Code  →  [Harvest]  →  [Synthetic Gen]  →  [QLoRA Train]  →  [GGUF Export]  →  [Ollama Deploy]
               500+ pairs    CoT + Neg samples    r=16, 3 epochs    Q4_K_M             metrobot
```

**Phase 1 — Dataset Builder** (`metrostack_dataset_builder.py`)

Walks your MetroStack project directory, identifies relevant source files (`.py`, `.ts`, `.sql`, `.yml`, `.dockerfile`, etc.), and optionally calls a local Ollama model to generate the instruction side of each training pair. Produces `metrostack_train.jsonl` and `metrostack_val.jsonl` in ChatML format.

**Phase 2 — Synthetic Generator** (`metrostack_synthetic_gen_v2.py`)

Turns your existing code snippets into 500+ training pairs using five mutation engines:

| Engine | What it produces |
|---|---|
| `instruct` | 3 distinct developer instructions per file → each with full CoT response |
| `mutate` | A structural variation (different route/table/prop/flag) |
| `debug` | Injects a subtle bug → "Fix this code" instruction pair |
| `docstring` | Strips comments → "Add documentation" instruction pair |
| `explain` | "What does this module do?" explanation pair |

Every output follows the mandatory **Think → Plan → Code** format:

```
## Reasoning
<Analyze constraints, edge cases, MetroStack environment requirements>

## Plan
<Numbered steps taken before writing code>

## Code
```python
<production-ready implementation>
```
```

Approximately 5% of the dataset is **negative samples** — refusal pairs that train MetroBot to reject dangerous or stack-incompatible requests (bulk data deletion, auth bypass, wrong framework substitution, secret exposure).

**Phase 3 — Fine-Tuning** (`metrobot_train.py`)

QLoRA fine-tuning via Unsloth on the generated dataset. Recommended configuration:

- Base model: `Qwen2.5-Coder-1.5B-Instruct` (best quality under 3B)
- LoRA rank: `r=16` for 1.5B models, `r=32` for 3B
- Epochs: 3
- Quantization: 4-bit base + bfloat16 activations
- Export: GGUF Q4_K_M (target: Snapdragon 8 / 16GB RAM)

**Phase 4 — Deployment** (`Modelfile`)

Ollama Modelfile with three critical inference parameters:
- `temperature 0.2` — Precision over creativity. MetroBot is an engineer, not a poet.
- `stop "<|im_end|>"` — Prevents ChatML control token bleed into responses.
- `SYSTEM` block — Reactivates the fine-tuned weights by reinstating environmental context.

**Phase 5 — Smoke Testing** (`metrobot_smoke_test.py`)

20 targeted prompts scored across 5 dimensions. Exits with code `0` (grade A/B) or `1` (grade C/F) for CI integration.

---

## Quick Start

### Requirements

```bash
pip install requests tiktoken tqdm colorama
```

For training (WSL2 / Linux with NVIDIA GPU):

```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Step 1 — Generate Training Data

```bash
# Using local Ollama as teacher (recommended — zero cost)
python metrostack_synthetic_gen_v2.py \
    --project C:/MetroStack \
    --model qwen2.5-coder:7b \
    --max-files 120 \
    --out ./dataset

# Output: dataset/metrostack_train.jsonl  (~500+ pairs)
#         dataset/metrostack_val.jsonl
#         dataset/synth_summary.json
```

### Step 2 — Fine-Tune (WSL2 / Linux)

```bash
python metrobot_train.py \
    --train ./dataset/metrostack_train.jsonl \
    --val   ./dataset/metrostack_val.jsonl \
    --base  unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit \
    --rank  16 \
    --epochs 3 \
    --quant q4_k_m \
    --out   ./metrobot_output

# Output: metrobot_output/merged_hf/         (HuggingFace checkpoint)
#         metrobot_output/gguf/metrobot-Q4_K_M.gguf
#         metrobot_output/gguf/Modelfile      (auto-generated)
```

### Step 3 — Deploy to Ollama

```bash
cd ./metrobot_output/gguf
ollama create metrobot -f Modelfile
ollama run metrobot
```

### Step 4 — Smoke Test

```bash
# Full 20-test suite
python metrobot_smoke_test.py

# Fast mode (8 critical tests)
python metrobot_smoke_test.py --fast

# Save results to disk
python metrobot_smoke_test.py --save-responses ./smoke_results
```

---

## Smoke Test Suite

20 tests across 6 categories, scored on 5 dimensions:

| Category | Tests | What's Verified |
|---|---|---|
| FastAPI / Backend | T01–T03 | Async routes, asyncpg deps, background tasks |
| PostGIS / pgpointcloud | T04–T06 | `PC_Explode`, `PC_Get`, `ST_DWithin`, DDL |
| Docker | T07–T08 | Compose V2 healthchecks, multi-stage Dockerfile |
| React / three.js / Zustand | T09–T10 | Vertex color shaders, state store patterns |
| Numerics / Point Cloud | T11–T12 | scipy KDE, Open3D + trimesh deviation math |
| Refusal / Safety | T14–T18 | Must refuse: data deletion, auth bypass, wrong stack |

**Scoring dimensions:**

| Dimension | Description |
|---|---|
| `cot_structure` | All three sections present: `## Reasoning`, `## Plan`, `## Code` |
| `forbidden` | Zero forbidden libraries in output |
| `env_compliance` | Correct stack-specific patterns referenced |
| `refusal_correct` | Dangerous prompts refused; valid prompts answered |
| `code_quality` | Non-empty, syntactically plausible code block present |

**Grading:** A ≥ 90% · B ≥ 75% · C ≥ 60% · F < 60%

---

## Interpreting Training Loss

| Loss | Meaning |
|---|---|
| < 0.5 | Excellent — model has internalized the patterns |
| 0.5 – 1.0 | Good — solid for production use |
| 1.0 – 1.5 | Acceptable — add more training data or increase epochs |
| > 1.5 | Needs work — check dataset quality or increase LoRA rank |

---

## Recommended Base Models

All under 3B parameters, available as Unsloth 4-bit pre-quantized checkpoints:

| Model | Parameters | Best For |
|---|---|---|
| `unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit` | 1.5B | **Recommended** — best quality under 2B, strong SQL and Python |
| `unsloth/deepseek-coder-1.3b-instruct-bnb-4bit` | 1.3B | Smallest footprint, strong SQL |
| `unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit` | 3B | Maximum quality — pushes the hardware limit |
| `unsloth/Phi-3-mini-4k-instruct-bnb-4bit` | 3.8B | Strong CoT reasoning |

---

## Hardware Targets

| Target | Spec | Recommended Model |
|---|---|---|
| Snapdragon 8 device | 12–16GB RAM | Qwen2.5-Coder-1.5B Q4_K_M (~1GB) |
| HP notebook 16GB RAM | CPU inference | Qwen2.5-Coder-1.5B Q4_K_M or Q8_0 |
| NVIDIA GPU (training) | 8GB+ VRAM | Any — 12GB+ for r=32 |

---

## API Usage

MetroBot runs as a standard Ollama model and is accessible via the Ollama REST API:

```bash
curl http://localhost:11434/api/generate -d '{
  "model": "metrobot",
  "prompt": "Write a FastAPI route to ingest a .e57 point cloud into pgpointcloud",
  "stream": false,
  "options": { "temperature": 0.2 }
}'
```

Integration with your MetroStack toolchain — pipe the response directly into your editor, a VS Code extension, or a local CLI tool.

---

## Design Decisions

**Why Chain-of-Thought?** Small models (sub-3B) benefit disproportionately from CoT training. Forcing the model to write a Reasoning and Plan section before the code block dramatically reduces hallucinated function names and incorrect PostGIS API usage.

**Why negative samples?** A model that doesn't know how to say no is dangerous in a production metrology context. Bulk-deleting scan history, bypassing auth, or substituting psycopg2 into async routes are all failure modes that must be explicitly trained out — not hoped away.

**Why Ollama over llama.cpp directly?** Ollama handles model lifecycle (load/unload), provides a REST API compatible with OpenAI clients, and makes model versioning (`ollama rm metrobot && ollama create metrobot`) trivial during iteration.

**Why ChatML format?** It's the current standard for Qwen, DeepSeek, and Mistral instruct models — the three model families best suited to the sub-3B parameter target. Alpaca format is retained as a fallback option in the dataset builder for LLaMA-family bases.

---

## Author

**Phillip May** — CMM Programmer, GD&T Specialist, MetroStack Architect  
Windsor, Ontario  
Project: MetroStack (internal metrology backend — not for redistribution)
