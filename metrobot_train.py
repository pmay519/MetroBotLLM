#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║          MetroBot  —  Unsloth QLoRA Fine-Tuning Script                 ║
# ║          Author : Phillip May / MetroStack Project                     ║
# ║          Target : <3B GGUF Q4_K_M / Q8_0  for Ollama / LM Studio      ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
metrobot_train.py
─────────────────────────────────────────────────────────────────────────────
Runs inside WSL2 (Ubuntu 22.04+) or a Linux machine with NVIDIA GPU.
CPU-only mode is also supported but very slow (use for smoke-testing only).

Recommended base models (pick ONE — uncomment your choice below):
  • Qwen2.5-Coder-1.5B-Instruct  ← BEST for Snapdragon 8 / 16GB RAM target
  • DeepSeek-Coder-1.3B-Instruct ← Smaller, fast, strong SQL
  • Phi-3-mini-4k-instruct        ← Microsoft, strong reasoning for size
  • Qwen2.5-Coder-3B-Instruct    ← Push the limit, still <3B

Hardware requirements:
  GPU training   : NVIDIA GPU, 8GB+ VRAM (12GB recommended for r=32)
  CPU fallback   : Any machine — slow but functional for smoke tests

Setup (WSL2 / Linux):
  pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
  pip install --no-deps trl peft accelerate bitsandbytes xformers
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

Usage:
  python metrobot_train.py --train ./metrostack_dataset/metrostack_train.jsonl \
                            --val   ./metrostack_dataset/metrostack_val.jsonl  \
                            --out   ./metrobot_output \
                            --rank  16
"""

import os, sys, json, argparse
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# ARG PARSE  (before heavy imports so --help is fast)
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MetroBot Unsloth QLoRA Fine-Tune")
    p.add_argument("--train",  type=Path, default=Path("./metrostack_dataset/metrostack_train.jsonl"))
    p.add_argument("--val",    type=Path, default=Path("./metrostack_dataset/metrostack_val.jsonl"))
    p.add_argument("--out",    type=Path, default=Path("./metrobot_output"))
    p.add_argument("--base",   type=str,
                   default="unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit",
                   help="HuggingFace model ID (Unsloth 4-bit pre-quantized preferred)")
    p.add_argument("--rank",   type=int, default=16,
                   help="LoRA rank r (16 for 1.5B, 32 for 3B+ models)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch",  type=int, default=2,
                   help="Per-device batch size (reduce to 1 if OOM)")
    p.add_argument("--grad-accum", type=int, default=4,
                   help="Gradient accumulation steps (effective batch = batch * grad_accum)")
    p.add_argument("--lr",     type=float, default=2e-4)
    p.add_argument("--max-seq", type=int, default=2048,
                   help="Max sequence length — must match your dataset token budget")
    p.add_argument("--quant",  choices=["q4_k_m", "q8_0", "f16"], default="q4_k_m",
                   help="GGUF quantization level for export")
    p.add_argument("--cpu-only", action="store_true",
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
    print(f"  Device : {DEVICE}")
    if HAS_CUDA:
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
except ImportError:
    print("[ERROR] PyTorch not found. Run the setup commands in the docstring.")
    sys.exit(1)

try:
    if HAS_CUDA:
        from unsloth import FastLanguageModel
        from unsloth.chat_templates import get_chat_template
        UNSLOTH = True
        print("  Unsloth : ✓")
    else:
        UNSLOTH = False
        print("  Unsloth : ✗ (CPU mode — using plain HF transformers)")
except ImportError:
    UNSLOTH = False
    print("  Unsloth : ✗ (not installed — using plain HF transformers for CPU fallback)")

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, TrainingArguments,
    DataCollatorForSeq2Seq,
)
try:
    from trl import SFTTrainer
except ImportError:
    print("[ERROR] trl not found. Run:  pip install trl")
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
        print(f"[WARN] Dataset not found: {path}")
        return []
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception:
                    pass
    return data

print(f"\n[MetroBot] Loading datasets...")
train_raw = load_jsonl(args.train)
val_raw   = load_jsonl(args.val)
print(f"  Train : {len(train_raw)} examples")
print(f"  Val   : {len(val_raw)} examples")

if not train_raw:
    print("[ERROR] No training data found. Run the synthetic generator first.")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOAD
# ─────────────────────────────────────────────────────────────────────────────

print(f"\n[MetroBot] Loading base model: {args.base}")

# ── Recommended base model options ──────────────────────────────────────────
# Uncomment the one you want, or pass via --base argument:
#
# BEST for <3B target (Snapdragon 8 / 16GB RAM):
#   "unsloth/Qwen2.5-Coder-1.5B-Instruct-bnb-4bit"
#
# Strong SQL, smallest footprint:
#   "unsloth/DeepSeek-Coder-V2-Lite-Instruct-bnb-4bit"  (too large — use below)
#   "unsloth/deepseek-coder-1.3b-instruct-bnb-4bit"
#
# Push the 3B limit:
#   "unsloth/Qwen2.5-Coder-3B-Instruct-bnb-4bit"
#
# Strong reasoning:
#   "unsloth/Phi-3-mini-4k-instruct-bnb-4bit"
# ────────────────────────────────────────────────────────────────────────────

if UNSLOTH and HAS_CUDA:
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name      = args.base,
        max_seq_length  = args.max_seq,
        dtype           = None,          # auto-detect: bfloat16 on Ampere+
        load_in_4bit    = True,          # QLoRA 4-bit base
    )

    # ── Apply LoRA adapters ──────────────────────────────────────────────────
    # r=16  : Good for 1.5B–3B models, spatial SQL, FastAPI patterns
    # r=32  : Better for complex CoT reasoning — needs more VRAM
    # alpha : Usually = rank (r) for stable training
    model = FastLanguageModel.get_peft_model(
        model,
        r               = args.rank,
        target_modules  = [
            "q_proj", "k_proj", "v_proj", "o_proj",      # Attention
            "gate_proj", "up_proj", "down_proj",           # MLP / FFN
        ],
        lora_alpha      = args.rank,
        lora_dropout    = 0.05,
        bias            = "none",
        use_gradient_checkpointing = "unsloth",  # Saves ~30% VRAM
        random_state    = 42,
        use_rslora      = False,
        loftq_config    = None,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params : {trainable:,}  ({100*trainable/total:.2f}% of {total:,})")

else:
    # CPU fallback — no QLoRA, just a smoke-test load
    print("  [CPU MODE] Loading in float32 — for smoke testing only")
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.float32)
    print("  [WARN] QLoRA / Unsloth not available on CPU. Training will be very slow.")

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZATION
# ─────────────────────────────────────────────────────────────────────────────

def format_chatml(example: dict) -> dict:
    """Convert ChatML messages list → single formatted string for SFT."""
    messages = example.get("messages", [])
    # Build ChatML string: <|im_start|>role\ncontent<|im_end|>\n
    text = ""
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += "<|im_start|>assistant\n"
    return {"text": text}

def tokenize(example: dict) -> dict:
    out = tokenizer(
        example["text"],
        truncation    = True,
        max_length    = args.max_seq,
        padding       = False,
        return_tensors = None,
    )
    out["labels"] = out["input_ids"].copy()
    return out

print(f"\n[MetroBot] Formatting and tokenizing datasets...")

train_formatted = [format_chatml(ex) for ex in train_raw]
val_formatted   = [format_chatml(ex) for ex in val_raw] if val_raw else train_formatted[:10]

train_ds = Dataset.from_list(train_formatted)
val_ds   = Dataset.from_list(val_formatted)

train_tok = train_ds.map(tokenize, batched=False, remove_columns=["text"])
val_tok   = val_ds.map(tokenize,   batched=False, remove_columns=["text"])

print(f"  Train tokenized : {len(train_tok)} sequences")
print(f"  Val tokenized   : {len(val_tok)} sequences")

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
    logging_steps               = 10,
    eval_strategy               = "steps",
    eval_steps                  = 50,
    save_strategy               = "steps",
    save_steps                  = 100,
    save_total_limit            = 2,
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    report_to                   = "none",         # Set "wandb" if you want W&B tracking
    seed                        = 42,
    dataloader_num_workers      = 0,              # 0 = main process (WSL2 safe)
    group_by_length             = True,           # Speeds up training, reduces padding
    optim                       = "adamw_8bit" if HAS_CUDA else "adamw_torch",
)

# ─────────────────────────────────────────────────────────────────────────────
# TRAINER
# ─────────────────────────────────────────────────────────────────────────────

data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model           = model,
    pad_to_multiple_of = 8,
    return_tensors  = "pt",
    padding         = True,
)

trainer = SFTTrainer(
    model           = model,
    tokenizer       = tokenizer,
    args            = training_args,
    train_dataset   = train_tok,
    eval_dataset    = val_tok,
    data_collator   = data_collator,
    dataset_text_field = None,         # We pre-tokenized
    max_seq_length  = args.max_seq,
    packing         = True,            # Packs short sequences together → higher GPU util
)

print(f"""
[MetroBot] Training Configuration
  Base model   : {args.base}
  LoRA rank    : r={args.rank}  alpha={args.rank}  dropout=0.05
  Epochs       : {args.epochs}
  Batch size   : {args.batch} × {args.grad_accum} grad_accum = {args.batch * args.grad_accum} effective
  LR           : {args.lr}  (cosine schedule)
  Max seq len  : {args.max_seq}
  Precision    : {'bfloat16' if HAS_CUDA and torch.cuda.is_bf16_supported() else 'float16' if HAS_CUDA else 'float32 (CPU)'}
  Output       : {args.out}/
""")

print("[MetroBot] Starting training — watch for loss < 1.0 within first epoch...")
print("  TIP: Loss should decrease steadily. If it plateaus above 1.5, increase epochs.")
print("  TIP: If OOM, reduce --batch to 1 and increase --grad-accum to 8.\n")

trainer_stats = trainer.train()

print(f"\n[MetroBot] Training complete!")
print(f"  Final loss     : {trainer_stats.training_loss:.4f}")
print(f"  Runtime        : {trainer_stats.metrics.get('train_runtime', 0):.0f}s")
print(f"  Samples/sec    : {trainer_stats.metrics.get('train_samples_per_second', 0):.2f}")

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
    quant_map = {"q4_k_m": "q4_k_m", "q8_0": "q8_0", "f16": "f16"}
    quant = quant_map.get(args.quant, "q4_k_m")

    print(f"\n[MetroBot] Converting to GGUF ({quant.upper()}) ...")

    if UNSLOTH and HAS_CUDA:
        # Unsloth's built-in GGUF conversion (easiest path)
        model.save_pretrained_gguf(
            str(gguf_path / "metrobot"),
            tokenizer,
            quantization_method = quant,
        )
        gguf_file = gguf_path / f"metrobot-{quant.upper()}.gguf"
        print(f"  ✓ GGUF saved: {gguf_file}")
    else:
        # Manual path: use llama.cpp convert script
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

  # 3. Or use llama-quantize for post-conversion quant:
  #    ./llama.cpp/llama-quantize {gguf_path}/metrobot-f16.gguf \\
  #        {gguf_path}/metrobot-Q4_K_M.gguf Q4_K_M
        """)
        gguf_file = gguf_path / f"metrobot-{quant.upper()}.gguf"

    # Write Modelfile alongside GGUF for direct Ollama import
    modelfile_content = f"""FROM ./{gguf_file.name}

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
║                    MetroBot Training Complete                   ║
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

Loss Interpretation:
  < 0.5  : Excellent — model has internalized the patterns
  0.5–1.0: Good — solid for production use
  1.0–1.5: Acceptable — add more training data or epochs
  > 1.5  : Needs work — check dataset quality or increase rank
""")
