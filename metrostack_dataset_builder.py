#!/usr/bin/env python3
"""
metrostack_dataset_builder.py
─────────────────────────────────────────────────────────────────────────────
MetroStack Fine-Tune Dataset Builder
Author  : Phillip May / MetroStack Project
Purpose : Harvest project source files and produce a JSONL instruction-response
          dataset for fine-tuning a <3B GGUF code model (e.g. DeepSeek-Coder-1.3B,
          Qwen2.5-Coder-1.5B-Instruct) using Unsloth + QLoRA.

Phases
──────
1. HARVEST   — Walk project directory, collect target source files.
2. ENRICH    — Call a "teacher" LLM (local Ollama or API) to generate the
               instruction side of each pair from the raw code.
3. MANUAL    — Load a hand-crafted seed file (metrostack_seeds.jsonl) of your
               best verified examples.
4. VALIDATE  — Token-length filter, dedup, shuffle, train/val split.
5. EXPORT    — Write metrostack_train.jsonl + metrostack_val.jsonl

Requirements (pip install):
    requests tiktoken tqdm colorama pathspec

Usage:
    python metrostack_dataset_builder.py --project C:/MetroStack \
        --teacher ollama \
        --ollama-model qwen2.5-coder:7b \
        --seed-file metrostack_seeds.jsonl \
        --out-dir ./dataset \
        --max-tokens 2048
"""

import os
import sys
import json
import time
import hashlib
import argparse
import random
import textwrap
from pathlib import Path
from typing import Generator

try:
    import requests
    from tqdm import tqdm
    from colorama import Fore, Style, init as colorama_init
    import tiktoken
except ImportError:
    print("[ERROR] Missing deps. Run:  pip install requests tiktoken tqdm colorama")
    sys.exit(1)

colorama_init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

METROSTACK_FILE_EXTENSIONS = {
    ".py", ".ts", ".tsx", ".js", ".jsx",         # Python / TypeScript / React
    ".sql",                                        # PostgreSQL / PostGIS queries
    ".json", ".jsonc",                             # Config / schema
    ".yaml", ".yml",                               # Docker Compose / k8s
    ".toml",                                       # pyproject.toml, etc.
    ".sh", ".bash",                                # Shell scripts
    ".dockerfile", ".Dockerfile",                  # Docker
    ".env.example",                                # Env templates (NOT .env!)
    ".md",                                         # Architecture docs
}

# Directories to skip entirely
SKIP_DIRS = {
    "node_modules", ".git", "__pycache__", ".venv", "venv",
    ".next", "dist", "build", ".mypy_cache", ".pytest_cache",
    "migrations",   # Alembic auto-migrations can pollute the dataset
}

# Files to always skip
SKIP_FILES = {
    ".env", ".env.local", ".env.production",       # Never include secrets!
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml",
}

# MetroStack-specific keywords — files containing these get PRIORITY boost
METROSTACK_KEYWORDS = [
    "pointcloud", "point_cloud", "pgpointcloud", "postgis", "fastapi",
    "open3d", "trimesh", "scipy", "numpy", "metraSCAN", "inspection",
    "docker-compose", "dockerfile", "zustand", "three.js", "three",
    "postgresql", "asyncpg", "sqlalchemy", "alembic", "uvicorn",
    "deviation", "tolerance", "gd&t", "cad", "scan", "mesh",
]

# Maximum file size to ingest (bytes) — skip huge auto-generated files
MAX_FILE_BYTES = 64_000

# Alphanumeric hash for dedup
def file_hash(content: str) -> str:
    return hashlib.md5(content.encode()).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 1 — HARVEST
# ─────────────────────────────────────────────────────────────────────────────

def harvest_files(project_root: Path) -> list[dict]:
    """Walk the project and return a list of {path, content, priority} dicts."""
    collected = []
    skipped = 0

    print(f"\n{Fore.CYAN}[PHASE 1]{Style.RESET_ALL} Harvesting source files from: {project_root}\n")

    for root, dirs, files in os.walk(project_root):
        # Prune directories in-place so os.walk doesn't descend into them
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]

        for filename in files:
            fpath = Path(root) / filename

            # Extension filter
            suffix = fpath.suffix.lower()
            # Handle "Dockerfile" (no extension)
            if filename in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml"):
                suffix = ".dockerfile"

            if suffix not in METROSTACK_FILE_EXTENSIONS:
                skipped += 1
                continue

            if filename in SKIP_FILES:
                print(f"  {Fore.RED}[SKIP-SECRET]{Style.RESET_ALL} {fpath.name}")
                skipped += 1
                continue

            if fpath.stat().st_size > MAX_FILE_BYTES:
                print(f"  {Fore.YELLOW}[SKIP-LARGE]{Style.RESET_ALL} {fpath.name} ({fpath.stat().st_size // 1024}KB)")
                skipped += 1
                continue

            try:
                content = fpath.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as e:
                print(f"  {Fore.RED}[READ-ERR]{Style.RESET_ALL} {fpath}: {e}")
                skipped += 1
                continue

            if len(content) < 80:  # Skip near-empty stubs
                skipped += 1
                continue

            # Priority: files containing MetroStack-specific terms rank higher
            content_lower = content.lower()
            priority = sum(1 for kw in METROSTACK_KEYWORDS if kw in content_lower)

            collected.append({
                "path": str(fpath),
                "filename": filename,
                "ext": suffix,
                "content": content,
                "priority": priority,
                "hash": file_hash(content),
            })

    # Sort by priority descending — best files first
    collected.sort(key=lambda x: x["priority"], reverse=True)

    print(f"  {Fore.GREEN}Collected:{Style.RESET_ALL} {len(collected)} files")
    print(f"  {Fore.YELLOW}Skipped:  {Style.RESET_ALL} {skipped} files")

    return collected


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 — ENRICH (Teacher LLM)
# ─────────────────────────────────────────────────────────────────────────────

TEACHER_SYSTEM_PROMPT = textwrap.dedent("""\
    You are a senior software architect specializing in metrology backends.
    The stack is: FastAPI, PostgreSQL 15 + PostGIS + pgpointcloud, React,
    three.js, Zustand, Open3D, trimesh, scipy, numpy, Docker / docker-compose.
    
    Given a source file from the "MetroStack" project, write a concise,
    specific one-to-two sentence INSTRUCTION that describes EXACTLY what this
    code does and what a developer would ask for to produce it.
    
    Rules:
    - Be specific. Mention the route, table, function, or component by name.
    - Do NOT include any code in your response.
    - Do NOT say "This code..." — phrase it as a developer instruction.
    - Output ONLY the instruction text, nothing else.
    
    Example output:
    "Create a FastAPI POST route /ingest/pointcloud that accepts a multipart
     .e57 file upload, parses it with Open3D, and bulk-inserts the XYZ points
     into the PostGIS pgpointcloud table scan_patches using asyncpg."
""")

def build_instruction_ollama(content: str, filename: str, model: str, base_url: str) -> str | None:
    """Call a local Ollama model to generate the instruction."""
    prompt = f"File: {filename}\n\n```\n{content[:8000]}\n```"
    payload = {
        "model": model,
        "system": TEACHER_SYSTEM_PROMPT,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 200},
    }
    try:
        r = requests.post(f"{base_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        print(f"    {Fore.RED}[OLLAMA-ERR]{Style.RESET_ALL} {e}")
        return None


def build_instruction_anthropic(content: str, filename: str, api_key: str, model: str = "claude-opus-4-5") -> str | None:
    """Call Anthropic API to generate the instruction (costs tokens — use sparingly)."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": 300,
        "system": TEACHER_SYSTEM_PROMPT,
        "messages": [
            {"role": "user", "content": f"File: {filename}\n\n```\n{content[:8000]}\n```"}
        ],
    }
    try:
        r = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=60)
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    except Exception as e:
        print(f"    {Fore.RED}[API-ERR]{Style.RESET_ALL} {e}")
        return None


def enrich_files(
    files: list[dict],
    teacher: str,
    ollama_model: str,
    ollama_url: str,
    anthropic_key: str | None,
    limit: int,
) -> list[dict]:
    """Generate instruction-output pairs for the top `limit` files."""
    pairs = []
    candidates = files[:limit]

    print(f"\n{Fore.CYAN}[PHASE 2]{Style.RESET_ALL} Enriching {len(candidates)} files with teacher LLM ({teacher})\n")

    for item in tqdm(candidates, desc="  Enriching", unit="file"):
        instruction = None

        if teacher == "ollama":
            instruction = build_instruction_ollama(
                item["content"], item["filename"], ollama_model, ollama_url
            )
        elif teacher == "anthropic" and anthropic_key:
            instruction = build_instruction_anthropic(
                item["content"], item["filename"], anthropic_key
            )
            time.sleep(0.5)  # Rate-limit courtesy

        if instruction and len(instruction) > 20:
            pairs.append({
                "instruction": instruction,
                "input": "",
                "output": item["content"],
                "source": item["filename"],
                "priority": item["priority"],
            })
        else:
            # Fallback: generate a generic instruction from filename + extension
            ext_map = {
                ".py": "Python module", ".sql": "SQL query", ".yml": "Docker Compose config",
                ".yaml": "YAML config", ".ts": "TypeScript module", ".tsx": "React component",
            }
            kind = ext_map.get(item["ext"], "source file")
            pairs.append({
                "instruction": f"Write the MetroStack {kind} for `{item['filename']}`.",
                "input": "",
                "output": item["content"],
                "source": item["filename"],
                "priority": item["priority"],
            })

    print(f"  {Fore.GREEN}Generated {len(pairs)} instruction-output pairs{Style.RESET_ALL}")
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 3 — SEED (Hand-crafted examples)
# ─────────────────────────────────────────────────────────────────────────────

def load_seeds(seed_file: Path) -> list[dict]:
    """Load hand-crafted JSONL seed examples."""
    if not seed_file.exists():
        print(f"\n{Fore.YELLOW}[PHASE 3]{Style.RESET_ALL} No seed file found at {seed_file} — skipping.")
        return []

    seeds = []
    print(f"\n{Fore.CYAN}[PHASE 3]{Style.RESET_ALL} Loading seed examples from {seed_file}")
    with open(seed_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "instruction" in obj and "output" in obj:
                    obj.setdefault("input", "")
                    obj.setdefault("source", "seed")
                    obj.setdefault("priority", 10)  # Seeds are highest priority
                    seeds.append(obj)
                else:
                    print(f"  {Fore.YELLOW}[WARN]{Style.RESET_ALL} Line {line_num}: missing instruction or output — skipped")
            except json.JSONDecodeError as e:
                print(f"  {Fore.RED}[ERROR]{Style.RESET_ALL} Line {line_num}: {e}")

    print(f"  {Fore.GREEN}Loaded {len(seeds)} seed examples{Style.RESET_ALL}")
    return seeds


# ─────────────────────────────────────────────────────────────────────────────
# BUILT-IN SEED EXAMPLES (MetroStack bootstrap — edit these!)
# ─────────────────────────────────────────────────────────────────────────────

BUILTIN_SEEDS = [
    {
        "instruction": "Create a FastAPI POST route /api/v1/scans/ingest that accepts a multipart E57 point cloud file, validates it exists, and returns the scan_id and point count.",
        "input": "",
        "output": textwrap.dedent("""\
            from fastapi import APIRouter, UploadFile, File, HTTPException
            from app.db import get_db
            import uuid, tempfile, os

            router = APIRouter(prefix="/api/v1/scans", tags=["scans"])

            @router.post("/ingest")
            async def ingest_scan(file: UploadFile = File(...), db=Depends(get_db)):
                if not file.filename.endswith(".e57"):
                    raise HTTPException(status_code=400, detail="Only .e57 files are accepted.")
                scan_id = str(uuid.uuid4())
                with tempfile.NamedTemporaryFile(delete=False, suffix=".e57") as tmp:
                    tmp.write(await file.read())
                    tmp_path = tmp.name
                try:
                    import open3d as o3d
                    pcd = o3d.io.read_point_cloud(tmp_path)
                    point_count = len(pcd.points)
                    # TODO: insert into pgpointcloud patches table
                    return {"scan_id": scan_id, "point_count": point_count, "status": "ingested"}
                finally:
                    os.unlink(tmp_path)
        """),
        "source": "builtin_seed",
        "priority": 10,
    },
    {
        "instruction": "Write a PostgreSQL function that queries pgpointcloud patch data and returns XYZ deviation statistics (min, max, mean, stddev) for a given scan_id.",
        "input": "",
        "output": textwrap.dedent("""\
            CREATE OR REPLACE FUNCTION get_scan_deviation_stats(p_scan_id UUID)
            RETURNS TABLE(
                min_dev FLOAT8, max_dev FLOAT8,
                mean_dev FLOAT8, stddev_dev FLOAT8, point_count BIGINT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT
                    MIN(PC_Get(pt, 'deviation'))::FLOAT8,
                    MAX(PC_Get(pt, 'deviation'))::FLOAT8,
                    AVG(PC_Get(pt, 'deviation'))::FLOAT8,
                    STDDEV(PC_Get(pt, 'deviation'))::FLOAT8,
                    COUNT(*)::BIGINT
                FROM (
                    SELECT PC_Explode(pa) AS pt
                    FROM scan_patches
                    WHERE scan_id = p_scan_id
                ) exploded;
            END;
            $$ LANGUAGE plpgsql;
        """),
        "source": "builtin_seed",
        "priority": 10,
    },
    {
        "instruction": "Write a docker-compose.yml for the MetroStack backend with services for PostgreSQL 15 + PostGIS + pgpointcloud, FastAPI app, and a Redis cache.",
        "input": "",
        "output": textwrap.dedent("""\
            version: "3.9"
            services:
              db:
                image: postgis/postgis:15-3.4
                environment:
                  POSTGRES_USER: metrostack
                  POSTGRES_PASSWORD: ${DB_PASSWORD}
                  POSTGRES_DB: metrostack
                volumes:
                  - pgdata:/var/lib/postgresql/data
                  - ./db/init:/docker-entrypoint-initdb.d
                ports:
                  - "5432:5432"
                healthcheck:
                  test: ["CMD-SHELL", "pg_isready -U metrostack"]
                  interval: 10s
                  retries: 5

              api:
                build: ./backend
                env_file: .env
                ports:
                  - "8000:8000"
                depends_on:
                  db:
                    condition: service_healthy
                command: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

              redis:
                image: redis:7-alpine
                ports:
                  - "6379:6379"

            volumes:
              pgdata:
        """),
        "source": "builtin_seed",
        "priority": 10,
    },
    {
        "instruction": "Create a Zustand store in TypeScript for managing the active 3D scan viewer state, including the loaded scan ID, point cloud visibility, and deviation color map range.",
        "input": "",
        "output": textwrap.dedent("""\
            import { create } from 'zustand';

            interface ScanViewerState {
              activeScanId: string | null;
              showPointCloud: boolean;
              deviationRange: [number, number]; // [min_mm, max_mm]
              setActiveScan: (id: string | null) => void;
              togglePointCloud: () => void;
              setDeviationRange: (range: [number, number]) => void;
              resetViewer: () => void;
            }

            export const useScanViewerStore = create<ScanViewerState>((set) => ({
              activeScanId: null,
              showPointCloud: true,
              deviationRange: [-0.5, 0.5],
              setActiveScan: (id) => set({ activeScanId: id }),
              togglePointCloud: () => set((s) => ({ showPointCloud: !s.showPointCloud })),
              setDeviationRange: (range) => set({ deviationRange: range }),
              resetViewer: () => set({ activeScanId: null, showPointCloud: true, deviationRange: [-0.5, 0.5] }),
            }));
        """),
        "source": "builtin_seed",
        "priority": 10,
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 4 — VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def validate_and_deduplicate(pairs: list[dict], max_tokens: int) -> list[dict]:
    """Filter by token length, deduplicate by output hash."""
    print(f"\n{Fore.CYAN}[PHASE 4]{Style.RESET_ALL} Validating and deduplicating {len(pairs)} pairs\n")

    try:
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        enc = None
        print(f"  {Fore.YELLOW}[WARN]{Style.RESET_ALL} tiktoken unavailable — skipping token-length filter")

    seen_hashes = set()
    valid = []
    too_long = 0
    dupes = 0

    for pair in pairs:
        combined = pair["instruction"] + pair["output"]
        h = file_hash(combined)

        if h in seen_hashes:
            dupes += 1
            continue
        seen_hashes.add(h)

        if enc:
            token_count = len(enc.encode(combined))
            if token_count > max_tokens:
                too_long += 1
                continue

        valid.append(pair)

    print(f"  {Fore.GREEN}Valid:    {Style.RESET_ALL} {len(valid)}")
    print(f"  {Fore.YELLOW}Dupes:    {Style.RESET_ALL} {dupes}")
    print(f"  {Fore.YELLOW}Too long: {Style.RESET_ALL} {too_long}")
    return valid


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 5 — EXPORT
# ─────────────────────────────────────────────────────────────────────────────

ALPACA_FORMAT = "alpaca"  # {"instruction", "input", "output"}
CHATML_FORMAT  = "chatml"  # ChatML <|im_start|> format for Qwen/Mistral

def export_dataset(
    pairs: list[dict],
    out_dir: Path,
    val_split: float,
    fmt: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    # Shuffle and split
    random.shuffle(pairs)
    split_idx = max(1, int(len(pairs) * (1 - val_split)))
    train = pairs[:split_idx]
    val   = pairs[split_idx:]

    print(f"\n{Fore.CYAN}[PHASE 5]{Style.RESET_ALL} Exporting dataset ({fmt} format)")
    print(f"  Train: {len(train)} | Val: {len(val)}")

    def to_chatml(pair: dict) -> dict:
        messages = [
            {"role": "system", "content": "You are MetroBot, a specialized MetroStack backend engineer. Answer with precise, production-ready code."},
            {"role": "user",   "content": pair["instruction"]},
            {"role": "assistant", "content": pair["output"]},
        ]
        return {"messages": messages}

    def write_jsonl(path: Path, data: list[dict], formatter):
        with open(path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(formatter(item), ensure_ascii=False) + "\n")
        print(f"  {Fore.GREEN}Wrote:{Style.RESET_ALL} {path} ({len(data)} examples)")

    if fmt == ALPACA_FORMAT:
        formatter = lambda p: {"instruction": p["instruction"], "input": p.get("input", ""), "output": p["output"]}
    else:
        formatter = to_chatml

    write_jsonl(out_dir / "metrostack_train.jsonl", train, formatter)
    write_jsonl(out_dir / "metrostack_val.jsonl",   val,   formatter)

    # Also write a human-readable summary
    summary = {
        "total_pairs": len(pairs),
        "train": len(train),
        "val": len(val),
        "format": fmt,
        "sources": list({p.get("source", "unknown") for p in pairs}),
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  {Fore.GREEN}Summary:{Style.RESET_ALL} {out_dir / 'dataset_summary.json'}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="MetroStack Fine-Tune Dataset Builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--project",      type=Path, default=Path("."),
                   help="Root of your MetroStack project directory")
    p.add_argument("--teacher",      choices=["ollama", "anthropic", "none"], default="none",
                   help="Teacher LLM to generate instructions (default: none = fallback filenames)")
    p.add_argument("--ollama-model", default="qwen2.5-coder:7b",
                   help="Ollama model name for teacher enrichment")
    p.add_argument("--ollama-url",   default="http://localhost:11434",
                   help="Ollama server base URL")
    p.add_argument("--anthropic-key", default=os.getenv("ANTHROPIC_API_KEY"),
                   help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    p.add_argument("--seed-file",    type=Path, default=Path("metrostack_seeds.jsonl"),
                   help="Path to hand-crafted seed JSONL file")
    p.add_argument("--out-dir",      type=Path, default=Path("./metrostack_dataset"),
                   help="Output directory for JSONL files")
    p.add_argument("--max-files",    type=int,  default=500,
                   help="Max source files to enrich (default: 500)")
    p.add_argument("--max-tokens",   type=int,  default=2048,
                   help="Max tokens per training pair (default: 2048)")
    p.add_argument("--val-split",    type=float, default=0.1,
                   help="Fraction of data to use for validation (default: 0.1)")
    p.add_argument("--format",       choices=["alpaca", "chatml"], default="chatml",
                   help="Output format: 'alpaca' for LLaMA, 'chatml' for Qwen/Mistral (default: chatml)")
    p.add_argument("--no-builtin-seeds", action="store_true",
                   help="Skip the built-in MetroStack seed examples")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║        MetroStack Fine-Tune Dataset Builder             ║
║        Target: <3B GGUF Q4K_M  |  Unsloth + QLoRA      ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    all_pairs: list[dict] = []

    # ── Phase 1: Harvest ──────────────────────────────────────────────────────
    if args.project.exists():
        files = harvest_files(args.project)
    else:
        print(f"{Fore.YELLOW}[WARN]{Style.RESET_ALL} Project path not found: {args.project} — skipping harvest")
        files = []

    # ── Phase 2: Enrich ───────────────────────────────────────────────────────
    if files and args.teacher != "none":
        enriched = enrich_files(
            files,
            teacher=args.teacher,
            ollama_model=args.ollama_model,
            ollama_url=args.ollama_url,
            anthropic_key=args.anthropic_key,
            limit=args.max_files,
        )
        all_pairs.extend(enriched)
    elif files:
        # No teacher — use fallback filename-based instructions
        print(f"\n{Fore.CYAN}[PHASE 2]{Style.RESET_ALL} No teacher selected — using filename-based instructions for {min(len(files), args.max_files)} files\n")
        ext_map = {
            ".py": "Python module", ".sql": "SQL file", ".yml": "YAML config",
            ".yaml": "YAML config", ".ts": "TypeScript module", ".tsx": "React component",
            ".dockerfile": "Dockerfile", ".sh": "shell script", ".md": "documentation file",
        }
        for item in files[:args.max_files]:
            kind = ext_map.get(item["ext"], "source file")
            all_pairs.append({
                "instruction": f"Write the MetroStack {kind} `{item['filename']}` with correct implementations for the stack.",
                "input": "",
                "output": item["content"],
                "source": item["filename"],
                "priority": item["priority"],
            })
        print(f"  {Fore.GREEN}Generated {len(all_pairs)} pairs (filename-based){Style.RESET_ALL}")

    # ── Phase 3: Seeds ────────────────────────────────────────────────────────
    if not args.no_builtin_seeds:
        print(f"\n{Fore.CYAN}[PHASE 3]{Style.RESET_ALL} Adding {len(BUILTIN_SEEDS)} built-in seed examples")
        all_pairs.extend(BUILTIN_SEEDS)

    seed_pairs = load_seeds(args.seed_file)
    all_pairs.extend(seed_pairs)

    if not all_pairs:
        print(f"\n{Fore.RED}[ERROR]{Style.RESET_ALL} No pairs collected. Check --project path or add seed examples.")
        sys.exit(1)

    # ── Phase 4: Validate ─────────────────────────────────────────────────────
    valid_pairs = validate_and_deduplicate(all_pairs, args.max_tokens)

    if len(valid_pairs) < 10:
        print(f"\n{Fore.YELLOW}[WARN]{Style.RESET_ALL} Only {len(valid_pairs)} valid pairs — consider adding more source files or seeds.")

    # ── Phase 5: Export ───────────────────────────────────────────────────────
    export_dataset(valid_pairs, args.out_dir, args.val_split, args.format)

    print(f"""
{Fore.GREEN}══════════════════════════════════════════════════════════
 DONE!  Dataset ready in: {args.out_dir}
══════════════════════════════════════════════════════════{Style.RESET_ALL}

Next Steps:
  1. Review metrostack_train.jsonl — spot-check 10 random entries
  2. Add more hand-crafted examples to metrostack_seeds.jsonl
  3. Fine-tune with Unsloth:
       pip install unsloth
       # Use the Unsloth Colab notebook for Qwen2.5-Coder-1.5B-Instruct
  4. Export to GGUF Q4K_M:
       llama.cpp: python convert_hf_to_gguf.py ./output_model --outtype q4_k_m
  5. Drop into Ollama:
       ollama create metrobot -f ./Modelfile
       ollama run metrobot "Write a PostGIS query for scan deviation stats"
""")


if __name__ == "__main__":
    main()
