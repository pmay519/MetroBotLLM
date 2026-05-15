#!/usr/bin/env python3
"""
metrostack_synthetic_gen.py
─────────────────────────────────────────────────────────────────────────────
MetroStack Synthetic Training Data Generator
Author  : Phillip May / MetroStack Project
Purpose : Turn YOUR existing code snippets into 500+ JSONL training pairs
          using a local Ollama model — zero API cost, runs fully offline.

Strategy (5 mutation engines per snippet):
  1. EXPLAIN    — "What does this code do?" → description pair
  2. INSTRUCT   — Generate 3 distinct instructions that would produce this code
  3. MUTATE     — Ask Ollama to write a variation (different table/route/param)
  4. DEBUG      — Inject a subtle bug, pair: "Fix this code" → fixed version
  5. DOCSTRING  — Strip comments/docstrings, pair: "Add docs" → original

This means 1 snippet → up to 7 pairs (3 from INSTRUCT + 1 each from rest).
200 snippets → 1,400+ pairs.  Even 75 snippets → 525+ pairs.

Requirements:
    pip install requests tqdm colorama tiktoken

Usage:
    # Point at your project and let it rip:
    python metrostack_synthetic_gen.py --project C:/MetroStack --out ./dataset

    # Or point at a folder of .py/.sql/.ts snippets:
    python metrostack_synthetic_gen.py --snippets C:/MetroStack/snippets --out ./dataset

    # Control the model and batch size:
    python metrostack_synthetic_gen.py --project C:/MetroStack \
        --model qwen2.5-coder:7b --max-files 150 --out ./dataset
"""

import os, sys, json, time, hashlib, argparse, random, re, textwrap
from pathlib import Path

try:
    import requests
    from tqdm import tqdm
    from colorama import Fore, Style, init as colorama_init
    import tiktoken
except ImportError:
    print("[ERROR] Run:  pip install requests tqdm colorama tiktoken")
    sys.exit(1)

colorama_init(autoreset=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

TARGET_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".sql", ".yml", ".yaml", ".sh", ".md"}
SKIP_DIRS  = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".next", "dist", "build"}
SKIP_FILES = {".env", ".env.local", "package-lock.json", "yarn.lock"}
MAX_FILE_BYTES = 48_000
MIN_FILE_CHARS = 120

METROSTACK_CONTEXT = """You are an expert in the MetroStack metrology backend project.
The stack is: FastAPI (Python), PostgreSQL 15 + PostGIS + pgpointcloud, React + TypeScript,
three.js + react-three-fiber, Zustand state management, Open3D, trimesh, scipy, numpy,
Docker / docker-compose, asyncpg, SQLAlchemy, Alembic.
The domain is industrial 3D metrology: point clouds, scan deviation analysis, GD&T, CAD comparison."""

OLLAMA_URL  = "http://localhost:11434"
OLLAMA_OPTS = {"temperature": 0.7, "num_predict": 600, "stop": ["```\n\n", "---END---"]}

enc = None
try:
    enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    pass

def token_count(s: str) -> int:
    return len(enc.encode(s)) if enc else len(s) // 4

def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def ollama(system: str, prompt: str, model: str, temperature: float = 0.7) -> str | None:
    opts = dict(OLLAMA_OPTS)
    opts["temperature"] = temperature
    payload = {"model": model, "system": system, "prompt": prompt,
                "stream": False, "options": opts}
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MUTATION ENGINES
# ─────────────────────────────────────────────────────────────────────────────

def engine_explain(code: str, filename: str, model: str) -> list[dict]:
    """Generate a 'what does this do' explanation pair."""
    system = METROSTACK_CONTEXT + "\nRespond with a concise technical explanation (3-6 sentences). No code."
    prompt = f"Explain what this MetroStack file does:\n\nFile: {filename}\n```\n{code[:3000]}\n```"
    result = ollama(system, prompt, model, temperature=0.3)
    if not result or len(result) < 40:
        return []
    return [{"instruction": f"Explain what the MetroStack file `{filename}` does.",
             "input": "", "output": result, "engine": "explain"}]


def engine_instruct(code: str, filename: str, model: str) -> list[dict]:
    """Generate 3 distinct developer instructions that would produce this code."""
    system = METROSTACK_CONTEXT + textwrap.dedent("""
        Given a code snippet, write EXACTLY 3 distinct developer instructions that would
        cause a coding assistant to produce this code. Each instruction should be phrased
        differently (imperative, question, specification style). Output them as a JSON array
        of strings ONLY. No markdown fences, no explanation. Example:
        ["Create a...", "How do I...", "Write a FastAPI route that..."]
    """)
    prompt = f"File: {filename}\n```\n{code[:3000]}\n```"
    raw = ollama(system, prompt, model, temperature=0.8)
    if not raw:
        return []
    # Strip markdown fences if model added them
    raw = re.sub(r"```[a-z]*\n?", "", raw).strip()
    try:
        instructions = json.loads(raw)
        if not isinstance(instructions, list):
            raise ValueError
        pairs = []
        for inst in instructions[:3]:
            if isinstance(inst, str) and len(inst) > 15:
                pairs.append({"instruction": inst.strip(), "input": "",
                               "output": code, "engine": "instruct"})
        return pairs
    except Exception:
        # Fallback: split by newline if JSON parse fails
        lines = [l.strip().strip('"').strip("'") for l in raw.split("\n") if len(l.strip()) > 15]
        return [{"instruction": l, "input": "", "output": code, "engine": "instruct"}
                for l in lines[:3]]


def engine_mutate(code: str, filename: str, model: str) -> list[dict]:
    """Ask the model to write a variation with different params/table/route name."""
    ext = Path(filename).suffix.lower()
    mutation_hints = {
        ".py":  "Change the route path, function name, or add an optional query parameter.",
        ".sql": "Change the table name or add a WHERE clause filtering by a different column.",
        ".ts":  "Change the component name and add one additional prop.",
        ".tsx": "Change the component name and add one additional prop.",
        ".yml": "Add a new service or change a port mapping.",
        ".sh":  "Add an error check or a different flag.",
    }
    hint = mutation_hints.get(ext, "Write a functionally similar but structurally different version.")
    system = METROSTACK_CONTEXT + f"\n{hint}\nOutput ONLY the modified code. No explanation."
    prompt = f"Mutate this MetroStack code:\n```\n{code[:3000]}\n```"
    result = ollama(system, prompt, model, temperature=0.9)
    if not result or len(result) < 60:
        return []
    # Clean code fences
    result = re.sub(r"^```[a-z]*\n?", "", result).rstrip("`").strip()
    instruction = f"Write a variation of the MetroStack `{filename}` that {hint.lower()}"
    return [{"instruction": instruction, "input": "", "output": result, "engine": "mutate"}]


def engine_debug(code: str, filename: str, model: str) -> list[dict]:
    """Inject a subtle bug, create a 'fix this' pair."""
    system = METROSTACK_CONTEXT + textwrap.dedent("""
        Introduce ONE subtle bug into the following code. The bug should be realistic:
        off-by-one error, wrong variable name, missing await, incorrect SQL column name,
        wrong HTTP method, missing return statement, etc.
        Output ONLY the buggy code. No explanation, no markdown fences.
    """)
    prompt = f"```\n{code[:2500]}\n```"
    buggy = ollama(system, prompt, model, temperature=0.6)
    if not buggy or len(buggy) < 60 or buggy == code:
        return []
    buggy = re.sub(r"^```[a-z]*\n?", "", buggy).rstrip("`").strip()
    instruction = (f"The following MetroStack code in `{filename}` has a subtle bug. "
                   f"Find and fix it:\n\n```\n{buggy[:2000]}\n```")
    return [{"instruction": instruction, "input": buggy, "output": code, "engine": "debug"}]


def engine_docstring(code: str, filename: str, model: str) -> list[dict]:
    """Strip comments/docstrings, create an 'add documentation' pair."""
    # Strip Python docstrings and # comments
    stripped = re.sub(r'"""[\s\S]*?"""', '""""""', code)
    stripped = re.sub(r"'''[\s\S]*?'''", "''''''", stripped)
    stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
    # Strip TS/JS comments
    stripped = re.sub(r"//.*$", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"/\*[\s\S]*?\*/", "", stripped)
    stripped = "\n".join(l for l in stripped.splitlines() if l.strip())

    if len(stripped) < 80 or stripped == code:
        return []

    instruction = (f"Add comprehensive docstrings and inline comments to this "
                   f"MetroStack `{filename}` file:\n\n```\n{stripped[:2000]}\n```")
    return [{"instruction": instruction, "input": stripped, "output": code, "engine": "docstring"}]


ENGINES = [engine_explain, engine_instruct, engine_mutate, engine_debug, engine_docstring]


# ─────────────────────────────────────────────────────────────────────────────
# FILE HARVESTER
# ─────────────────────────────────────────────────────────────────────────────

def harvest(paths: list[Path], max_files: int) -> list[dict]:
    found = []
    for root_path in paths:
        if root_path.is_file():
            if root_path.stat().st_size <= MAX_FILE_BYTES:
                try:
                    c = root_path.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(c) >= MIN_FILE_CHARS:
                        found.append({"path": str(root_path), "filename": root_path.name,
                                       "ext": root_path.suffix.lower(), "content": c})
                except Exception:
                    pass
            continue

        for r, dirs, files in os.walk(root_path):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for fn in files:
                fpath = Path(r) / fn
                if fn in SKIP_FILES:
                    continue
                sfx = fpath.suffix.lower()
                if fn in ("Dockerfile", "docker-compose.yml", "docker-compose.yaml"):
                    sfx = ".yml"
                if sfx not in TARGET_EXTENSIONS:
                    continue
                if fpath.stat().st_size > MAX_FILE_BYTES:
                    continue
                try:
                    c = fpath.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(c) >= MIN_FILE_CHARS:
                        found.append({"path": str(fpath), "filename": fn,
                                       "ext": sfx, "content": c})
                except Exception:
                    pass
    random.shuffle(found)
    return found[:max_files]


# ─────────────────────────────────────────────────────────────────────────────
# DEDUP + EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def dedup(pairs: list[dict], max_tokens: int) -> list[dict]:
    seen, out = set(), []
    for p in pairs:
        h = md5(p["instruction"] + p["output"])
        if h in seen:
            continue
        if token_count(p["instruction"] + p["output"]) > max_tokens:
            continue
        seen.add(h)
        out.append(p)
    return out


def to_chatml(p: dict) -> dict:
    return {"messages": [
        {"role": "system",    "content": "You are MetroBot, a specialized MetroStack engineer. Respond with precise, production-ready code and explanations."},
        {"role": "user",      "content": p["instruction"]},
        {"role": "assistant", "content": p["output"]},
    ]}


def export(pairs: list[dict], out_dir: Path, val_split: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_split)))
    train, val = pairs[:split], pairs[split:]

    def write(path, data):
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(to_chatml(p), ensure_ascii=False) + "\n")
        print(f"  {Fore.GREEN}→ {path}{Style.RESET_ALL}  ({len(data)} examples)")

    write(out_dir / "metrostack_train.jsonl", train)
    write(out_dir / "metrostack_val.jsonl",   val)

    # Engine breakdown stats
    stats: dict[str, int] = {}
    for p in pairs:
        stats[p.get("engine", "unknown")] = stats.get(p.get("engine", "unknown"), 0) + 1

    summary = {"total": len(pairs), "train": len(train), "val": len(val),
                "engine_breakdown": stats}
    (out_dir / "synth_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Engine breakdown: {stats}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MetroStack Synthetic Training Data Generator")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--project",  type=Path, help="Root of MetroStack project directory")
    src.add_argument("--snippets", type=Path, help="Folder of hand-picked snippet files")
    p.add_argument("--model",      default="qwen2.5-coder:7b",
                   help="Ollama model for generation (default: qwen2.5-coder:7b)")
    p.add_argument("--max-files",  type=int,  default=100,
                   help="Max source files to process (default: 100)")
    p.add_argument("--max-tokens", type=int,  default=2048,
                   help="Max tokens per pair (default: 2048)")
    p.add_argument("--val-split",  type=float, default=0.1,
                   help="Validation split fraction (default: 0.1)")
    p.add_argument("--out",        type=Path,  default=Path("./metrostack_dataset"),
                   help="Output directory")
    p.add_argument("--engines",    nargs="+",
                   choices=["explain", "instruct", "mutate", "debug", "docstring"],
                   default=["explain", "instruct", "mutate", "debug", "docstring"],
                   help="Which mutation engines to run")
    p.add_argument("--skip-ollama-check", action="store_true",
                   help="Skip Ollama connectivity check at startup")
    return p.parse_args()


def check_ollama(model: str):
    print(f"  Checking Ollama at {OLLAMA_URL}...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        available = [m for m in models if model.split(":")[0] in m]
        if available:
            print(f"  {Fore.GREEN}✓ Found model: {available[0]}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.YELLOW}⚠ Model '{model}' not found. Available: {models}")
            print(f"    Run:  ollama pull {model}{Style.RESET_ALL}")
            sys.exit(1)
    except Exception as e:
        print(f"  {Fore.RED}✗ Ollama not reachable: {e}")
        print(f"    Make sure Ollama is running:  ollama serve{Style.RESET_ALL}")
        sys.exit(1)


def main():
    args = parse_args()

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════╗
║     MetroStack Synthetic Training Data Generator        ║
║     5 Mutation Engines  |  Target: 500+ pairs           ║
╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    if not args.skip_ollama_check:
        check_ollama(args.model)

    # Resolve source paths
    source_paths = []
    if args.project:
        source_paths = [args.project]
    elif args.snippets:
        source_paths = [args.snippets]
    else:
        print(f"{Fore.YELLOW}No --project or --snippets specified. Using current directory.{Style.RESET_ALL}")
        source_paths = [Path(".")]

    # Harvest
    print(f"\n{Fore.CYAN}[HARVEST]{Style.RESET_ALL} Scanning source files...")
    files = harvest(source_paths, args.max_files)
    if not files:
        print(f"{Fore.RED}No eligible files found. Check your path.{Style.RESET_ALL}")
        sys.exit(1)
    print(f"  Found {Fore.GREEN}{len(files)}{Style.RESET_ALL} files → "
          f"estimated {Fore.GREEN}{len(files) * len(args.engines) * 2}+{Style.RESET_ALL} pairs\n")

    # Engine map
    engine_map = {
        "explain":   engine_explain,
        "instruct":  engine_instruct,
        "mutate":    engine_mutate,
        "debug":     engine_debug,
        "docstring": engine_docstring,
    }
    active_engines = [engine_map[e] for e in args.engines]

    # Generate
    all_pairs: list[dict] = []
    failed = 0

    print(f"{Fore.CYAN}[GENERATE]{Style.RESET_ALL} Running {len(active_engines)} engines on {len(files)} files...\n")

    with tqdm(total=len(files), desc="  Files", unit="file", colour="cyan") as pbar:
        for file in files:
            code     = file["content"]
            filename = file["filename"]
            file_pairs = []

            for engine_fn in active_engines:
                try:
                    results = engine_fn(code, filename, args.model)
                    file_pairs.extend(results)
                except Exception as e:
                    failed += 1

            all_pairs.extend(file_pairs)
            pbar.set_postfix({"pairs": len(all_pairs), "failed": failed})
            pbar.update(1)

    print(f"\n  {Fore.GREEN}Raw pairs generated: {len(all_pairs)}{Style.RESET_ALL}")
    print(f"  {Fore.YELLOW}Engine failures:     {failed}{Style.RESET_ALL}")

    # Dedup + filter
    print(f"\n{Fore.CYAN}[VALIDATE]{Style.RESET_ALL} Deduplicating and filtering...")
    clean = dedup(all_pairs, args.max_tokens)
    print(f"  {Fore.GREEN}Clean pairs: {len(clean)}{Style.RESET_ALL} "
          f"({len(all_pairs) - len(clean)} removed)")

    if len(clean) < 50:
        print(f"\n{Fore.YELLOW}[WARN] Only {len(clean)} pairs — consider more files or enabling all engines.{Style.RESET_ALL}")

    # Export
    print(f"\n{Fore.CYAN}[EXPORT]{Style.RESET_ALL} Writing dataset to {args.out}/")
    export(clean, args.out, args.val_split)

    target_hit = len(clean) >= 500
    color = Fore.GREEN if target_hit else Fore.YELLOW
    status = "TARGET HIT ✓" if target_hit else f"Need {500 - len(clean)} more pairs"

    print(f"""
{color}══════════════════════════════════════════════════════════
  {status}
  Total pairs: {len(clean)}
  Output:      {args.out}/
══════════════════════════════════════════════════════════{Style.RESET_ALL}

Tip: If you need more pairs, run again with:
  --engines instruct mutate   (highest variety)
  --max-files 200             (more source files)
  --model qwen2.5-coder:14b   (better quality if you have VRAM)
""")


if __name__ == "__main__":
    main()
