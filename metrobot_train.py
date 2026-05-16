#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MetroBot — Unsloth QLoRA Fine-Tuning Script  (v2 — FIXED)             ║
# ║  Author : Phillip May / MetroStack Project                              ║
# ║  Target : <3B GGUF Q4_K_M / Q8_0 for Ollama / LM Studio               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
metrobot_train.py  (v2 — corrected)
─────────────────────────────────────────────────────────────────────────────
Fixes applied vs v1:
  FIX 1 — format_chatml no longer appends a bare <|im_start|>assistant turn.
           All messages (including the assistant response) are already in the
           messages[] list from the JSONL.  Every turn now gets a closing
           <|im_end|> so the model learns where responses end.

  FIX 2 — packing=True conflict resolved.  SFTTrainer now receives raw-text
           datasets and handles tokenisation internally via dataset_text_field.
           The manual tokenize() map + DataCollatorForSeq2Seq are removed.

  FIX 3 — EOS token appended to every formatted sample so the model learns
           to terminate cleanly (prevents runaway/gibberish generation).

  FIX 4 — eval_steps / save_steps guardrailed: they are clamped to never
           exceed the number of training steps so HF Trainer doesn't crash on
           small datasets.

  FIX 5 — lora_alpha set to 2× rank (standard best-practice for instruct
           fine-tuning) rather than == rank, giving better gradient scaling.

Runs inside WSL2 (Ubuntu 22.04+) or a Linux machine with NVIDIA GPU.
CPU-only mode is supported for smoke-testing only.

Recommended base models (pick ONE — or pass via --base):
  • unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit  ← BEST for <2B / Snapdragon
  • unsloth/deepseek-coder-1.3b-instruct-bnb-4bit  ← Smallest footprint, strong SQL
  • unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit     ← Max quality, pushes hardware limit
  • unsloth/Phi-3-mini-4k-instruct-bnb-4bit         ← Strong CoT reasoning

Hardware:
  GPU training  : NVIDIA GPU, 8 GB+ VRAM (12 GB recommended for r=32)
  CPU fallback  : Any machine — slow, for smoke tests only

Setup (WSL2 / Linux):
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes xformers
  pip install torch --index-url https://download.pytorch.org/whl/cu121

Usage:
  python metrobot_train.py \\
      --train ./dataset/metrostack_train.jsonl \\
      --val   ./dataset/metrostack_val.jsonl   \\
      --out   ./metrobot_output                \\
      --rank  16 --epochs 5
"""

import os, sys, json, argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSE (before heavy imports so --help is fast)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MetroBot Unsloth QLoRA Fine-Tune (v2)")
    p.add_argument("--train",     type=Path,  default=Path("./dataset/metrostack_train.jsonl"))
    p.add_argument("--val",       type=Path,  default=Path("./dataset/metrostack_val.jsonl"))
    p.add_argument("--out",       type=Path,  default=Path("./metrobot_output"))
    p.add_argument("--base",      type=str,   default="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
                   help="HuggingFace model ID (Unsloth 4-bit pre-quantized preferred)")
    p.add_argument("--rank",      type=int,   default=16,
                   help="LoRA rank r (16 for 1.5B, 32 for 3B+ models)")
    p.add_argument("--epochs",    type=int,   default=5,
                   help="Training epochs (5–7 recommended for CoT format learning)")
    p.add_argument("--batch",     type=int,   default=2,
                   help="Per-device batch size (reduce to 1 if OOM)")
    p.add_argument("--grad-accum",type=int,   default=4,
                   help="Gradient accumulation steps (effective batch = batch × grad_accum)")
    p.add_argument("--lr",        type=float, default=2e-4)
    p.add_argument("--max-seq",   type=int,   default=2048,
                   help="Max sequence length — must match your dataset token budget")
    p.add_argument("--quant",     choices=["q4_k_m", "q8_0", "f16"], default="q4_k_m",
                   help="GGUF quantisation level for export")
    p.add_argument("--cpu-only",  action="store_true",
                   help="Force CPU training (smoke test only — extremely slow)")
    p.add_argument("--skip-gguf", action="store_true",
                   help="Skip GGUF conversion (keep HF checkpoint only)")
    return p.parse_args()

args = parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

print("\n[MetroBot] Loading libraries...")

try:
    import torch
    HAS_CUDA = torch.cuda.is_available() and not args.cpu_only
    DEVICE   = "cuda" if HAS_CUDA else "cpu"
    print(f"  Device  : {DEVICE}")
    if HAS_CUDA:
        print(f"  GPU     : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM    : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("[ERROR] PyTorch not found. Run the setup commands in the docstring.")
    sys.exit(1)

UNSLOTH = False
try:
    if HAS_CUDA:
        from unsloth import FastLanguageModel
        UNSLOTH = True
        print("  Unsloth : ✓")
    else:
        print("  Unsloth : ✗ (CPU mode — using plain HF transformers)")
except ImportError:
    print("  Unsloth : ✗ (not installed — falling back to plain HF transformers)")

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
)
try:
    from trl import SFTTrainer, SFTConfig
except ImportError:
    print("[ERROR] trl not found. Run: pip install trl")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# METROSTACK SYSTEM PROMPT  (must match synthetic_gen_v2.py exactly)
# ─────────────────────────────────────────────────────────────────────────────

METROBOT_SYSTEM = (
    "You are MetroBot, a senior industrial metrology software engineer specialized in the "
    "MetroStack project. You write production-ready code for 3D scan processing pipelines.\n\n"
    "Environment: FastAPI 0.111 / Python 3.12, PostgreSQL 16 + PostGIS 3.4 + pgpointcloud 1.2, "
    "asyncpg 0.29, SQLAlchemy 2.x async, React 18 + TypeScript 5 + Capacitor 6, "
    "three.js r165 + react-three-fiber v8, Zustand 4.x, Docker Compose V2, "
    "Open3D 0.18, trimesh 4.x, numpy 1.26, scipy 1.13.\n\n"
    "FORBIDDEN: psycopg2 sync, Redux, moment.js, Flask, SQLite, MySQL.\n\n"
    "Always respond with:\n"
    "## Reasoning\n<analysis>\n\n## Plan\n<steps>\n\n## Code\n```<lang>\n<code>\n```"
)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        print(f"  [WARN] Dataset not found: {path}")
        return []
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  [WARN] Skipping malformed line {i+1}: {e}")
    return data

print(f"\n[MetroBot] Loading datasets...")
train_raw = load_jsonl(args.train)
val_raw   = load_jsonl(args.val)
print(f"  Train   : {len(train_raw)} examples")
print(f"  Val     : {len(val_raw)} examples")

if not train_raw:
    print("[ERROR] No training data found. Run the dataset validator and synthetic generator first.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[MetroBot] Loading base model: {args.base}")

if UNSLOTH and HAS_CUDA:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name   = args.base,
        max_seq_length = args.max_seq,
        dtype        = None,   # auto-detect: bfloat16 on Ampere+
        load_in_4bit = True,   # QLoRA 4-bit base
    )

    # FIX 5: lora_alpha = rank * 2  (better gradient scaling for instruct tuning)
    lora_alpha = args.rank * 2
    model = FastLanguageModel.get_peft_model(
        model,
        r              = args.rank,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",   # Attention
            "gate_proj", "up_proj", "down_proj",        # MLP / FFN
        ],
        lora_alpha     = lora_alpha,
        lora_dropout   = 0.05,
        bias           = "none",
        use_gradient_checkpointing = "unsloth",  # Saves ~30% VRAM
        random_state   = 42,
        use_rslora     = False,
        loftq_config   = None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,} ({100*trainable/total:.2f}% of {total:,})")
    print(f"  LoRA alpha       : {lora_alpha}  (rank × 2)")

else:
    print("  [CPU MODE] Loading in float32 — for smoke testing only")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model     = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    print("  [WARN] QLoRA / Unsloth not available on CPU. Training will be very slow.")

# Ensure padding token is set (required for batched training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# ─────────────────────────────────────────────────────────────────────────────
# FIX 1 + FIX 3: CORRECT ChatML FORMATTING
#
# v1 bug: appended a bare <|im_start|>assistant\n after all messages,
#         doubling the assistant turn and never closing it with <|im_end|>.
#
# Fix:  Every message (including the assistant response) is already in
#       messages[].  We simply format them all — each gets its own
#       <|im_end|>.  We then append the tokenizer's EOS token so the
#       model learns where the full sample ends.
# ─────────────────────────────────────────────────────────────────────────────

def format_chatml(example: dict) -> dict:
    """
    Convert a ChatML messages list into a single formatted string for SFT.

    Expected JSONL structure:
        {
          "messages": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "## Reasoning\n...\n## Plan\n...\n## Code\n..."}
          ]
        }

    The assistant message MUST be the last entry and must contain the full
    CoT response.  Run metrobot_validate_dataset.py to verify this before
    training.
    """
    messages = example.get("messages", [])
    text = ""
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        # FIX 1: every turn — including assistant — gets its closing <|im_end|>
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"

    # FIX 3: append EOS so the model learns to terminate cleanly
    text += tokenizer.eos_token if tokenizer.eos_token else ""
    return {"text": text}

# ─────────────────────────────────────────────────────────────────────────────
# BUILD DATASETS  (raw text — SFTTrainer tokenises internally)
# FIX 2: we pass dataset_text_field="text" and let SFTTrainer handle
#         tokenisation + packing.  The manual tokenize() map and
#         DataCollatorForSeq2Seq are removed — they conflicted with packing.
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[MetroBot] Formatting datasets into ChatML...")
train_formatted = [format_chatml(ex) for ex in train_raw]
val_formatted   = [format_chatml(ex) for ex in val_raw] if val_raw else train_formatted[:max(1, len(train_formatted)//10)]

train_ds = Dataset.from_list(train_formatted)
val_ds   = Dataset.from_list(val_formatted)
print(f"  Train samples : {len(train_ds)}")
print(f"  Val samples   : {len(val_ds)}")

# Quick format sanity check on first sample
sample_text = train_ds[0]["text"]
assert "<|im_start|>assistant" in sample_text, \
    "[ERROR] First sample missing assistant turn — check your JSONL structure."
assert sample_text.count("<|im_end|>") >= 2, \
    "[ERROR] First sample has fewer than 2 <|im_end|> tokens — format_chatml is broken."
print("  Format check  : ✓ (im_end tokens present, assistant turn confirmed)")

# ─────────────────────────────────────────────────────────────────────────────
# FIX 4: GUARDRAIL eval_steps / save_steps vs dataset size
# ─────────────────────────────────────────────────────────────────────────────

effective_batch  = args.batch * args.grad_accum
steps_per_epoch  = max(1, len(train_ds) // effective_batch)
total_steps      = steps_per_epoch * args.epochs

# Never let eval/save steps exceed total training steps
eval_steps_safe  = min(50, max(1, steps_per_epoch))
save_steps_safe  = min(100, max(1, steps_per_epoch))

print(f"\n  Steps/epoch   : {steps_per_epoch}")
print(f"  Total steps   : {total_steps}")
print(f"  Eval every    : {eval_steps_safe} steps")
print(f"  Save every    : {save_steps_safe} steps")

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING ARGUMENTS
# ─────────────────────────────────────────────────────────────────────────────

args.out.mkdir(parents=True, exist_ok=True)

training_args = TrainingArguments(
    output_dir                  = str(args.out / "checkpoints"),
    num_train_epochs            = args.epochs,
    per_device_train_batch_size = args.batch,
    per_device_eval_batch_size  = 1,
    gradient_accumulation_steps = args.grad_accum,
    warmup_ratio                = 0.05,
    learning_rate               = args.lr,
    lr_scheduler_type           = "cosine",
    fp16                        = HAS_CUDA and not torch.cuda.is_bf16_supported(),
    bf16                        = HAS_CUDA and torch.cuda.is_bf16_supported(),
    logging_steps               = max(1, steps_per_epoch // 5),
    eval_strategy               = "steps",
    eval_steps                  = eval_steps_safe,   # FIX 4
    save_strategy               = "steps",
    save_steps                  = save_steps_safe,   # FIX 4
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    report_to                   = "none",            # set "wandb" if you want W&B tracking
    seed                        = 42,
    dataloader_num_workers      = 0,                 # 0 = main process (WSL2 safe)
    group_by_length             = True,              # reduces padding waste
    optim                       = "adamw_8bit" if HAS_CUDA else "adamw_torch",
)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# FIX 2: dataset_text_field="text" — SFTTrainer tokenises internally.
#         packing=True now works correctly because we're passing raw text.
#         DataCollatorForSeq2Seq is removed (was conflicting with packing).
# ─────────────────────────────────────────────────────────────────────────────

trainer = SFTTrainer(
    model              = model,
    tokenizer          = tokenizer,
    args               = training_args,
    train_dataset      = train_ds,
    eval_dataset       = val_ds,
    dataset_text_field = "text",   # FIX 2: point at raw text, let SFT handle it
    max_seq_length     = args.max_seq,
    packing            = True,     # FIX 2: now safe — SFT owns tokenisation
)

print(f"""
[MetroBot] Training Configuration
  Base model   : {args.base}
  LoRA rank    : r={args.rank}  alpha={args.rank * 2}  dropout=0.05
  Epochs       : {args.epochs}
  Batch size   : {args.batch} × {args.grad_accum} grad_accum = {effective_batch} effective
  LR           : {args.lr} (cosine schedule)
  Max seq len  : {args.max_seq}
  Precision    : {'bfloat16' if HAS_CUDA and torch.cuda.is_bf16_supported() else 'float16' if HAS_CUDA else 'float32 (CPU)'}
  Output       : {args.out}/
""")

print("[MetroBot] Starting training — target loss < 1.0 within first 2 epochs...")
print("  TIP: Loss should decrease steadily. If it plateaus above 1.5, increase --epochs.")
print("  TIP: If OOM, reduce --batch to 1 and increase --grad-accum to 8.\n")

trainer_stats = trainer.train()

print(f"\n[MetroBot] Training complete!")
print(f"  Final loss   : {trainer_stats.training_loss:.4f}")
print(f"  Runtime      : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
print(f"  Samples/sec  : {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")

# Loss interpretation
loss = trainer_stats.training_loss
if loss < 0.5:
    grade = "Excellent — model has fully internalized the patterns"
elif loss < 1.0:
    grade = "Good — solid for production use"
elif loss < 1.5:
    grade = "Acceptable — add more data or increase --epochs"
else:
    grade = "⚠ Needs work — check dataset quality, increase --rank or --epochs"
print(f"  Loss grade   : {grade}")

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MERGED MODEL (LoRA weights merged into base for GGUF conversion)
# ─────────────────────────────────────────────────────────────────────────────

merged_path = args.out / "merged_hf"
print(f"\n[MetroBot] Saving merged model to {merged_path} ...")

if UNSLOTH and HAS_CUDA:
    model.save_pretrained_merged(
        str(merged_path),
        tokenizer,
        save_method = "merged_16bit",   # Full precision merged weights
    )
else:
    model.save_pretrained(str(merged_path))
    tokenizer.save_pretrained(str(merged_path))

print(f"  ✓ Merged HF checkpoint saved")

# ─────────────────────────────────────────────────────────────────────────────
# GGUF EXPORT
# ─────────────────────────────────────────────────────────────────────────────

if not args.skip_gguf:
    gguf_path = args.out / "gguf"
    gguf_path.mkdir(exist_ok=True)

    quant = args.quant   # q4_k_m | q8_0 | f16

    print(f"\n[MetroBot] Converting to GGUF ({quant.upper()}) ...")

    if UNSLOTH and HAS_CUDA:
        model.save_pretrained_gguf(
            str(gguf_path / "metrobot"),
            tokenizer,
            quantization_method = quant,
        )
        gguf_file = gguf_path / f"metrobot-{quant.upper()}.gguf"
        print(f"  ✓ GGUF saved: {gguf_file}")
    else:
        print("  [WARN] Unsloth GGUF export not available (CPU mode).")
        print("  Run this manually after training:")
        print(f"""
  # 1. Clone llama.cpp (if not already):
  #    git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp
  #    pip install -r requirements.txt

  # 2. Convert merged HF → GGUF:
  python llama.cpp/convert_hf_to_gguf.py \\
      {merged_path} \\
      --outtype {quant} \\
      --outfile {gguf_path}/metrobot-{quant.upper()}.gguf
""")

    gguf_file_name = f"metrobot-{quant.upper()}.gguf"

    # Write Modelfile alongside GGUF for direct Ollama import
    modelfile_content = f"""FROM ./{gguf_file_name}

# ── MetroBot Identity ────────────────────────────────────────────────────────
SYSTEM \"\"\"
{METROBOT_SYSTEM}
\"\"\"

# ── Inference Parameters ─────────────────────────────────────────────────────
# Precision over creativity — this is an engineering assistant, not a poet.
PARAMETER temperature 0.2
PARAMETER top_p 0.85
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# Stop tokens for ChatML format (Qwen / DeepSeek instruct)
# These must match the tokens the model was trained to produce at turn end.
PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|endoftext|>"
"""
    modelfile_path = gguf_path / "Modelfile"
    modelfile_path.write_text(modelfile_content)
    print(f"  ✓ Modelfile saved: {modelfile_path}")

# ─────────────────────────────────────────────────────────────────────────────
# FINAL REPORT
# ─────────────────────────────────────────────────────────────────────────────

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  MetroBot Training Complete  (v2 — corrected)                   ║
╚══════════════════════════════════════════════════════════════════╝

  HF Checkpoint : {merged_path}/
  GGUF File     : {args.out}/gguf/metrobot-{args.quant.upper()}.gguf
  Modelfile     : {args.out}/gguf/Modelfile

Next Steps — Deploy to Ollama:
  1. cd {args.out}/gguf
  2. ollama create metrobot -f Modelfile
  3. ollama run metrobot "Write a PostGIS deviation query for scan_patches"

Smoke Test:
  ollama run metrobot \\
    "Create a FastAPI POST route to ingest a .e57 point cloud into pgpointcloud"

If output is still garbled after retraining:
  • Re-run metrobot_validate_dataset.py and fix any flagged JSONL pairs
  • Check eval_loss in checkpoints/ — target < 1.0 before deploying
  • Try --rank 32 with --epochs 7 for stronger CoT format adherence
  • Verify your Ollama model was recreated:
      ollama rm metrobot && ollama create metrobot -f Modelfile
""")
