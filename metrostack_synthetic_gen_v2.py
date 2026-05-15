#!/usr/bin/env python3
"""
metrostack_synthetic_gen_v2.py
─────────────────────────────────────────────────────────────────────────────
MetroStack Synthetic Training Data Generator  v2.0
Author  : Phillip May / MetroStack Project

Upgrades over v1:
  ✓ Chain-of-Thought (Think → Plan → Code) in every output
  ✓ Hardcoded MetroStack Environmental Constants in system prompt
  ✓ ChatML JSONL output (Unsloth / DeepSeek / Qwen instruct standard)
  ✓ Negative Sampling (~5% refusal/warning pairs for hallucination guard)
  ✓ 5 Mutation Engines: explain, instruct, mutate, debug, docstring

Output format per line:
  {"messages": [
    {"role": "system",    "content": "You are MetroBot..."},
    {"role": "user",      "content": "<instruction>"},
    {"role": "assistant", "content": "## Reasoning\n...\n\n## Plan\n...\n\n## Code\n```<lang>\n...\n```"}
  ]}

Requirements:
    pip install requests tqdm colorama tiktoken

Usage:
    python metrostack_synthetic_gen_v2.py --project C:/MetroStack --out ./dataset
    python metrostack_synthetic_gen_v2.py --snippets ./my_snippets --out ./dataset \
        --model qwen2.5-coder:7b --max-files 150
"""

import os, sys, json, re, time, hashlib, argparse, random, textwrap
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
# ❶  HARDCODED METROSTACK ENVIRONMENTAL CONSTANTS
#    These are injected into every system prompt so the model NEVER forgets
#    your specific stack. Edit this section to match your environment.
# ─────────────────────────────────────────────────────────────────────────────

METROSTACK_ENV = {
    "database":         "PostgreSQL 16 + PostGIS 3.4 + pgpointcloud 1.2",
    "db_driver":        "asyncpg 0.29 (never psycopg2 for async routes)",
    "orm":              "SQLAlchemy 2.x async + Alembic migrations",
    "backend":          "FastAPI 0.111 / Python 3.12 / Uvicorn",
    "frontend":         "React 18 + TypeScript 5 + Capacitor 6 (Android target)",
    "3d_rendering":     "three.js r165 + react-three-fiber v8 + drei v9",
    "state":            "Zustand 4.x (no Redux)",
    "containerization": "Docker Compose V2 (compose.yml, NOT docker-compose.yml)",
    "point_cloud_io":   "Open3D 0.18 for E57/PLY I/O, trimesh 4.x for mesh ops",
    "numerics":         "numpy 1.26, scipy 1.13",
    "spatial_fns":      "Use ST_* functions from PostGIS 3.4; PC_* from pgpointcloud",
    "auth":             "JWT via python-jose; API keys stored in environment variables",
    "forbidden":        "Never suggest: psycopg2 sync, Redux, moment.js, Flask, SQLite, MySQL",
}

SYSTEM_PROMPT = textwrap.dedent(f"""\
    You are MetroBot, a senior industrial metrology software engineer specialized in the
    MetroStack project. You write production-ready code for 3D scan processing pipelines.

    ## Your Environment (NEVER deviate from these):
    - Database   : {METROSTACK_ENV['database']}
    - DB Driver  : {METROSTACK_ENV['db_driver']}
    - ORM        : {METROSTACK_ENV['orm']}
    - Backend    : {METROSTACK_ENV['backend']}
    - Frontend   : {METROSTACK_ENV['frontend']}
    - 3D/Viewer  : {METROSTACK_ENV['3d_rendering']}
    - State Mgmt : {METROSTACK_ENV['state']}
    - Containers : {METROSTACK_ENV['containerization']}
    - Point Cloud: {METROSTACK_ENV['point_cloud_io']}
    - Numerics   : {METROSTACK_ENV['numerics']}
    - Spatial SQL: {METROSTACK_ENV['spatial_fns']}
    - Auth       : {METROSTACK_ENV['auth']}
    - FORBIDDEN  : {METROSTACK_ENV['forbidden']}

    ## Response Format (ALWAYS follow this structure):
    ## Reasoning
    <Analyze the requirement. What are the constraints? What could go wrong?>

    ## Plan
    <Step-by-step plan before writing a single line of code.>

    ## Code
    ```<language>
    <production-ready implementation>
    ```
""").strip()

# ─────────────────────────────────────────────────────────────────────────────
# ❷  NEGATIVE SAMPLE BANK  (~5% of dataset = refusal/warning pairs)
#    These teach MetroBot to refuse dangerous or impossible operations.
# ─────────────────────────────────────────────────────────────────────────────

NEGATIVE_SAMPLES = [
    {
        "instruction": "Write a SQL script to DELETE all rows from scan_patches and scans to reset the database for testing.",
        "output": textwrap.dedent("""\
            ## Reasoning
            This request asks to permanently delete all scan data and measurement history.
            In a metrology context, scan data is the source of truth for part inspection records,
            QA audits, and traceability. Irreversible bulk deletion is a critical data integrity risk.

            ## Plan
            I will not execute this operation directly. Instead I will explain the safe alternative.

            ## Response
            ⚠️ **REFUSED — Data Integrity Risk**

            Deleting all measurement history is a non-recoverable operation that violates metrology
            traceability requirements (ISO 10012, IATF 16949 records retention).

            **Safe alternative for test resets:**
            ```sql
            -- Use a dedicated test schema instead of truncating production tables
            CREATE SCHEMA IF NOT EXISTS test_metrostack;
            -- Or flag test records with a boolean column:
            ALTER TABLE scans ADD COLUMN IF NOT EXISTS is_test BOOLEAN DEFAULT FALSE;
            -- Delete ONLY test records:
            DELETE FROM scan_patches WHERE scan_id IN (SELECT scan_id FROM scans WHERE is_test = TRUE);
            DELETE FROM scans WHERE is_test = TRUE;
            ```
            Never run unscoped DELETE or TRUNCATE on production metrology data.
        """).strip(),
    },
    {
        "instruction": "Modify the MetroStack API to skip JWT authentication on all routes for easier development.",
        "output": textwrap.dedent("""\
            ## Reasoning
            Disabling authentication globally — even in development — creates a pattern that risks
            being accidentally deployed to production, exposing raw metrology data and scan archives.

            ## Plan
            Refuse the blanket disable. Provide a safe, scoped development alternative.

            ## Response
            ⚠️ **REFUSED — Security Risk**

            Globally disabling JWT authentication is not safe even in a dev environment.
            If this config leaks to production, all scan data and CMM reports are publicly accessible.

            **Safe dev alternative — use a dev-only dependency override:**
            ```python
            # app/deps.py
            import os
            from fastapi import Header, HTTPException

            async def get_current_user(authorization: str = Header(...)):
                if os.getenv("METROSTACK_ENV") == "development" and authorization == "Bearer dev-token":
                    return {"user_id": "dev", "role": "admin"}
                # ... real JWT validation for all other cases
                raise HTTPException(status_code=401, detail="Invalid token")
            ```
            Set `METROSTACK_ENV=development` only in your local `.env` and never commit it.
        """).strip(),
    },
    {
        "instruction": "Replace asyncpg with psycopg2 throughout the MetroStack backend for simplicity.",
        "output": textwrap.dedent("""\
            ## Reasoning
            psycopg2 is a synchronous driver. MetroStack's FastAPI backend is fully async.
            Mixing sync DB calls into async routes causes thread blocking, connection pool exhaustion,
            and will break under concurrent scan ingestion loads.

            ## Plan
            Refuse the substitution. Explain why asyncpg is non-negotiable and offer the correct path.

            ## Response
            ⚠️ **REFUSED — Architecture Incompatibility**

            psycopg2 is synchronous and **must not** be used in MetroStack's async FastAPI backend.
            Using it would block the event loop on every DB call, destroying throughput under load.

            **MetroStack's correct async stack:**
            ```python
            # Correct: asyncpg with SQLAlchemy 2.x async engine
            from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
            from sqlalchemy.orm import sessionmaker

            engine = create_async_engine(
                "postgresql+asyncpg://user:password@db:5432/metrostack",
                pool_size=10, max_overflow=20,
            )
            AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
            ```
            If you need a sync connection for a one-off migration script, use psycopg2 in isolation —
            never inside a FastAPI route handler.
        """).strip(),
    },
    {
        "instruction": "Store the JWT secret key directly in the FastAPI source code so it's easier to share with the team.",
        "output": textwrap.dedent("""\
            ## Reasoning
            Hardcoding secrets in source code causes them to be committed to version control,
            visible in Docker image layers, and leaked in bug reports. This is a critical
            security vulnerability for any production metrology system.

            ## Plan
            Refuse and demonstrate the correct pattern using environment variables.

            ## Response
            ⚠️ **REFUSED — Critical Security Violation**

            Never hardcode secrets in source files. Once committed to Git, a secret is compromised
            permanently — even if later removed, it lives in history.

            **Correct pattern:**
            ```python
            # app/config.py
            from pydantic_settings import BaseSettings

            class Settings(BaseSettings):
                jwt_secret_key: str          # Loaded from environment / .env file
                jwt_algorithm: str = "HS256"
                access_token_expire_minutes: int = 60

                class Config:
                    env_file = ".env"

            settings = Settings()
            ```
            Share secrets with your team via a secrets manager (Vault, AWS Secrets Manager)
            or an encrypted `.env` shared over a secure channel — never via source code.
        """).strip(),
    },
    {
        "instruction": "Write a FastAPI route that returns the raw .env file contents for debugging.",
        "output": textwrap.dedent("""\
            ## Reasoning
            Serving .env file contents over HTTP — even internally — exposes database passwords,
            JWT secrets, and API keys. This is a textbook security catastrophe.

            ## Plan
            Hard refuse. No safe alternative exists for this specific request.

            ## Response
            ⚠️ **REFUSED — Critical Secret Exposure**

            This route would broadcast your database credentials, JWT signing keys, and any
            third-party API tokens to anyone who can reach the endpoint. This cannot be made safe.

            **For debugging config values, expose only non-sensitive derived info:**
            ```python
            @router.get("/debug/config", include_in_schema=False)
            async def debug_config(current_user=Depends(require_admin)):
                return {
                    "db_host": settings.db_host,
                    "db_name": settings.db_name,
                    "environment": settings.environment,
                    # Never include: db_password, jwt_secret_key, api_keys
                }
            ```
            Restrict this route to admin users and disable it entirely in production.
        """).strip(),
    },
    {
        "instruction": "Use Flask instead of FastAPI for the MetroStack backend since Flask is simpler.",
        "output": textwrap.dedent("""\
            ## Reasoning
            Flask is synchronous by default and lacks native async support for the point cloud
            ingestion workloads MetroStack handles. Migrating would require rewriting all async
            DB calls, background tasks, and WebSocket endpoints.

            ## Plan
            Refuse the framework swap. Explain the architectural cost.

            ## Response
            ⚠️ **NOT RECOMMENDED — Architecture Regression**

            Flask is not a suitable replacement for FastAPI in MetroStack for these reasons:
            - Flask's sync model blocks on every I/O call (DB queries, file reads, Open3D ops)
            - FastAPI's async routes handle concurrent scan uploads without thread-pool overhead
            - FastAPI's Pydantic v2 models give you free request validation and OpenAPI docs
            - Flask-SQLAlchemy's sync ORM is incompatible with asyncpg connection pooling

            If simplicity is the goal, the correct move is improving FastAPI's project structure,
            not switching frameworks. MetroStack stays on FastAPI.
        """).strip(),
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TARGET_EXTENSIONS = {".py", ".ts", ".tsx", ".js", ".sql", ".yml", ".yaml", ".sh", ".md"}
SKIP_DIRS  = {"node_modules", ".git", "__pycache__", ".venv", "venv", ".next", "dist", "build"}
SKIP_FILES = {".env", ".env.local", "package-lock.json", "yarn.lock"}
MAX_FILE_BYTES = 48_000
MIN_FILE_CHARS = 120
OLLAMA_URL = "http://localhost:11434"

enc = None
try:
    enc = tiktoken.get_encoding("cl100k_base")
except Exception:
    pass

def token_count(s: str) -> int:
    return len(enc.encode(s)) if enc else len(s) // 4

def md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

# ─────────────────────────────────────────────────────────────────────────────
# ❸  OLLAMA CALLER  — instructs the model to use Think→Plan→Code format
# ─────────────────────────────────────────────────────────────────────────────

COT_SUFFIX = textwrap.dedent("""

    IMPORTANT: Your response MUST follow this exact structure:

    ## Reasoning
    <Analyze constraints, edge cases, MetroStack environment requirements>

    ## Plan
    <Numbered steps you will take before writing code>

    ## Code
    ```<language>
    <implementation>
    ```
""")

def ollama_cot(instruction: str, model: str, temperature: float = 0.7) -> str | None:
    """Call Ollama with CoT enforcement baked into the prompt."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT,
        "prompt": instruction + COT_SUFFIX,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 900, "stop": ["---END---"]},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=240)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return None

def ollama_raw(system_extra: str, prompt: str, model: str, temperature: float = 0.7) -> str | None:
    """Low-level call for intermediate steps (instruction generation, mutation hints)."""
    payload = {
        "model": model,
        "system": SYSTEM_PROMPT + "\n" + system_extra,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 400, "stop": ["---END---"]},
    }
    try:
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=180)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# ❹  MUTATION ENGINES  (each returns list of {instruction, output} dicts)
# ─────────────────────────────────────────────────────────────────────────────

def engine_instruct(code: str, filename: str, model: str) -> list[dict]:
    """Generate 3 instructions → each paired with CoT output for that instruction."""
    sys_extra = textwrap.dedent("""
        Given a MetroStack code file, output EXACTLY 3 developer instructions that would
        produce this code. Use different styles: imperative, question, specification.
        Output a raw JSON array of strings ONLY. No markdown, no explanation.
        Example: ["Create a...", "How do I...", "Write a route that..."]
    """)
    raw = ollama_raw(sys_extra, f"File: {filename}\n```\n{code[:3000]}\n```", model, temperature=0.8)
    if not raw:
        return []
    raw = re.sub(r"```[a-z]*\n?|```", "", raw).strip()
    try:
        instructions = json.loads(raw)
        if not isinstance(instructions, list):
            raise ValueError
    except Exception:
        instructions = [l.strip().strip('"\'') for l in raw.split("\n") if len(l.strip()) > 15][:3]

    pairs = []
    for inst in instructions[:3]:
        if not isinstance(inst, str) or len(inst) < 15:
            continue
        # Now generate a full CoT response for this instruction
        # (Use the original code as ground-truth but ask for CoT reasoning wrapper)
        cot_prompt = (
            f"{inst}\n\n"
            f"Reference implementation (use this as your code output, "
            f"but write genuine Reasoning and Plan sections):\n```\n{code[:2500]}\n```"
        )
        output = ollama_cot(cot_prompt, model, temperature=0.4)
        if output and "## Reasoning" in output and len(output) > 100:
            pairs.append({"instruction": inst, "output": output, "engine": "instruct"})
        else:
            # Fallback: wrap the code in a minimal CoT shell
            lang = _lang_from_filename(filename)
            pairs.append({
                "instruction": inst,
                "output": _cot_wrap(
                    reasoning=f"This is a MetroStack {lang} implementation for `{filename}`. "
                               f"Using {METROSTACK_ENV['database']} and the project's standard async patterns.",
                    plan="1. Implement the requested functionality.\n2. Follow MetroStack async conventions.\n3. Add error handling.",
                    lang=lang,
                    code=code,
                ),
                "engine": "instruct",
            })
    return pairs


def engine_mutate(code: str, filename: str, model: str) -> list[dict]:
    """Ask for a variation; wrap the variation in CoT."""
    ext = Path(filename).suffix.lower()
    hints = {
        ".py":  "Change the route path and add one optional query parameter with a default value.",
        ".sql": "Change the table alias and add a LIMIT clause with a configurable max rows parameter.",
        ".ts":  "Change the component name and add one additional typed prop with a sensible default.",
        ".tsx": "Change the component name and add one additional typed prop with a sensible default.",
        ".yml": "Add a healthcheck block to the main service and a named volume.",
        ".sh":  "Add a set -euo pipefail header and an error-trap function.",
    }
    hint = hints.get(ext, "Write a functionally similar but structurally different version.")
    sys_extra = f"Output ONLY the modified code. No explanation, no markdown fences. {hint}"
    raw = ollama_raw(sys_extra, f"```\n{code[:3000]}\n```", model, temperature=0.9)
    if not raw or len(raw) < 60:
        return []
    raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("`").strip()
    lang = _lang_from_filename(filename)
    instruction = f"Write a variation of the `{filename}` module that: {hint.lower()}"
    output = _cot_wrap(
        reasoning=f"Creating a variation of `{filename}`. Key change: {hint} "
                   f"Must remain compatible with {METROSTACK_ENV['database']} and the async FastAPI stack.",
        plan=f"1. Apply the requested structural change.\n2. Preserve all MetroStack conventions.\n3. Validate the output compiles correctly.",
        lang=lang,
        code=raw,
    )
    return [{"instruction": instruction, "output": output, "engine": "mutate"}]


def engine_debug(code: str, filename: str, model: str) -> list[dict]:
    """Inject a bug → 'Fix this' instruction → CoT output with fixed code."""
    sys_extra = textwrap.dedent("""
        Introduce ONE subtle realistic bug: off-by-one, wrong variable name, missing await,
        incorrect SQL column, wrong HTTP method, missing return, wrong PostGIS function.
        Output ONLY the buggy code. No explanation, no markdown fences.
    """)
    buggy = ollama_raw(sys_extra, f"```\n{code[:2500]}\n```", model, temperature=0.5)
    if not buggy or len(buggy) < 60 or buggy.strip() == code.strip():
        return []
    buggy = re.sub(r"^```[a-z]*\n?", "", buggy).rstrip("`").strip()
    lang = _lang_from_filename(filename)
    instruction = (
        f"The following MetroStack `{filename}` code has a subtle bug. "
        f"Identify it and provide the corrected version:\n\n```{lang}\n{buggy[:2000]}\n```"
    )
    output = _cot_wrap(
        reasoning=f"Analyzing the buggy `{filename}` code. Checking for: missing await keywords, "
                   f"incorrect asyncpg patterns, wrong PostGIS function usage, off-by-one errors, "
                   f"and type mismatches common in the MetroStack async stack.",
        plan="1. Read through the code carefully.\n2. Identify the introduced bug.\n3. Explain why it's a bug.\n4. Provide the corrected implementation.",
        lang=lang,
        code=code,
    )
    return [{"instruction": instruction, "output": output, "engine": "debug"}]


def engine_docstring(code: str, filename: str, model: str) -> list[dict]:
    """Strip comments → 'Add documentation' instruction → CoT output with documented code."""
    stripped = re.sub(r'"""[\s\S]*?"""', '""""""', code)
    stripped = re.sub(r"'''[\s\S]*?'''", "''''''", stripped)
    stripped = re.sub(r"^\s*#.*$", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"//.*$", "", stripped, flags=re.MULTILINE)
    stripped = re.sub(r"/\*[\s\S]*?\*/", "", stripped)
    stripped = "\n".join(l for l in stripped.splitlines() if l.strip())
    if len(stripped) < 80 or stripped == code:
        return []
    lang = _lang_from_filename(filename)
    instruction = (
        f"Add comprehensive docstrings, type hints, and inline comments to this "
        f"MetroStack `{filename}` file. Follow Google-style docstrings for Python, "
        f"JSDoc for TypeScript:\n\n```{lang}\n{stripped[:2000]}\n```"
    )
    output = _cot_wrap(
        reasoning=f"Adding documentation to `{filename}`. Will follow Google-style docstrings for Python "
                   f"or JSDoc for TypeScript. Must document: parameters, return types, exceptions, "
                   f"and MetroStack-specific context (PostGIS types, asyncpg patterns, etc.).",
        plan="1. Identify all public functions, classes, and exported components.\n"
              "2. Add module-level docstring with purpose and MetroStack context.\n"
              "3. Add per-function docstrings with Args, Returns, Raises.\n"
              "4. Add inline comments for non-obvious logic.",
        lang=lang,
        code=code,
    )
    return [{"instruction": instruction, "output": output, "engine": "docstring"}]


def engine_explain(code: str, filename: str, model: str) -> list[dict]:
    """Generate an explanation pair in CoT format."""
    sys_extra = "Write a concise technical explanation (4-8 sentences). No code in the response."
    explanation = ollama_raw(
        sys_extra,
        f"Explain what this MetroStack file does:\n\nFile: {filename}\n```\n{code[:3000]}\n```",
        model, temperature=0.3,
    )
    if not explanation or len(explanation) < 40:
        return []
    instruction = f"Explain what the MetroStack module `{filename}` does and how it fits the architecture."
    output = textwrap.dedent(f"""\
        ## Reasoning
        Analyzing `{filename}` in the context of the MetroStack architecture:
        - Backend: {METROSTACK_ENV['backend']}
        - Database: {METROSTACK_ENV['database']}
        - Containers: {METROSTACK_ENV['containerization']}

        ## Plan
        1. Describe the module's primary responsibility.
        2. Explain its dependencies within MetroStack.
        3. Note any MetroStack-specific patterns used.

        ## Explanation
        {explanation}
    """).strip()
    return [{"instruction": instruction, "output": output, "engine": "explain"}]


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _lang_from_filename(filename: str) -> str:
    return {
        ".py": "python", ".ts": "typescript", ".tsx": "typescript",
        ".js": "javascript", ".sql": "sql", ".yml": "yaml",
        ".yaml": "yaml", ".sh": "bash", ".md": "markdown",
    }.get(Path(filename).suffix.lower(), "text")

def _cot_wrap(reasoning: str, plan: str, lang: str, code: str) -> str:
    return textwrap.dedent(f"""\
        ## Reasoning
        {reasoning}

        ## Plan
        {plan}

        ## Code
        ```{lang}
        {code.strip()}
        ```
    """).strip()

ENGINES = {
    "instruct":  engine_instruct,
    "mutate":    engine_mutate,
    "debug":     engine_debug,
    "docstring": engine_docstring,
    "explain":   engine_explain,
}

# ─────────────────────────────────────────────────────────────────────────────
# HARVEST
# ─────────────────────────────────────────────────────────────────────────────

def harvest(paths: list[Path], max_files: int) -> list[dict]:
    found = []
    for root_path in paths:
        if root_path.is_file():
            try:
                c = root_path.read_text(encoding="utf-8", errors="ignore").strip()
                if len(c) >= MIN_FILE_CHARS and root_path.stat().st_size <= MAX_FILE_BYTES:
                    found.append({"filename": root_path.name, "ext": root_path.suffix.lower(), "content": c})
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
                if fn in ("Dockerfile", "compose.yml", "compose.yaml", "docker-compose.yml"):
                    sfx = ".yml"
                if sfx not in TARGET_EXTENSIONS:
                    continue
                if fpath.stat().st_size > MAX_FILE_BYTES:
                    continue
                try:
                    c = fpath.read_text(encoding="utf-8", errors="ignore").strip()
                    if len(c) >= MIN_FILE_CHARS:
                        found.append({"filename": fn, "ext": sfx, "content": c})
                except Exception:
                    pass
    random.shuffle(found)
    return found[:max_files]

# ─────────────────────────────────────────────────────────────────────────────
# DEDUP + VALIDATE
# ─────────────────────────────────────────────────────────────────────────────

def dedup_validate(pairs: list[dict], max_tokens: int) -> list[dict]:
    seen, out = set(), []
    for p in pairs:
        key = md5(p["instruction"] + p["output"])
        if key in seen:
            continue
        if token_count(p["instruction"] + p["output"]) > max_tokens:
            continue
        # Enforce CoT structure for non-negative pairs
        if p.get("engine") != "negative":
            if "## Reasoning" not in p["output"] or "## Plan" not in p["output"]:
                continue
        seen.add(key)
        out.append(p)
    return out

# ─────────────────────────────────────────────────────────────────────────────
# ❺  CHATML EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def to_chatml(p: dict) -> dict:
    return {"messages": [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": p["instruction"]},
        {"role": "assistant", "content": p["output"]},
    ]}

def export(pairs: list[dict], out_dir: Path, val_split: float):
    out_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * (1 - val_split)))
    train, val = pairs[:split], pairs[split:]

    def write(path: Path, data: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(to_chatml(p), ensure_ascii=False) + "\n")
        print(f"  {Fore.GREEN}→ {path}{Style.RESET_ALL}  ({len(data)} examples)")

    write(out_dir / "metrostack_train.jsonl", train)
    write(out_dir / "metrostack_val.jsonl",   val)

    stats: dict[str, int] = {}
    for p in pairs:
        e = p.get("engine", "unknown")
        stats[e] = stats.get(e, 0) + 1

    neg_count = stats.get("negative", 0)
    neg_pct   = round(neg_count / len(pairs) * 100, 1) if pairs else 0

    summary = {
        "total": len(pairs), "train": len(train), "val": len(val),
        "engine_breakdown": stats,
        "negative_sample_pct": f"{neg_pct}%  (target: ~5%)",
        "cot_enforced": True,
        "output_format": "ChatML",
        "env_constants": METROSTACK_ENV,
    }
    (out_dir / "synth_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  Engine breakdown : {stats}")
    print(f"  Negative samples : {neg_count} ({neg_pct}%)  target ≈ 5%")

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MetroStack Synthetic Data Generator v2")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--project",  type=Path)
    src.add_argument("--snippets", type=Path)
    p.add_argument("--model",      default="qwen2.5-coder:7b")
    p.add_argument("--max-files",  type=int,   default=100)
    p.add_argument("--max-tokens", type=int,   default=2048)
    p.add_argument("--val-split",  type=float, default=0.1)
    p.add_argument("--out",        type=Path,  default=Path("./metrostack_dataset"))
    p.add_argument("--engines",    nargs="+",
                   choices=list(ENGINES.keys()),
                   default=list(ENGINES.keys()))
    p.add_argument("--skip-ollama-check", action="store_true")
    return p.parse_args()

def check_ollama(model: str):
    print(f"  Checking Ollama at {OLLAMA_URL} ...")
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        names = [m["name"] for m in r.json().get("models", [])]
        match = [n for n in names if model.split(":")[0] in n]
        if match:
            print(f"  {Fore.GREEN}✓ Model ready: {match[0]}{Style.RESET_ALL}")
        else:
            print(f"  {Fore.YELLOW}⚠ '{model}' not found. Available: {names}")
            print(f"    Run:  ollama pull {model}{Style.RESET_ALL}")
            sys.exit(1)
    except Exception as e:
        print(f"  {Fore.RED}✗ Ollama unreachable: {e}\n    Start it with:  ollama serve{Style.RESET_ALL}")
        sys.exit(1)

def main():
    args = parse_args()

    print(f"""
{Fore.CYAN}╔══════════════════════════════════════════════════════════════╗
║   MetroStack Synthetic Data Generator  v2.0                 ║
║   CoT Enforced │ ChatML │ Neg Sampling │ Env Constants      ║
╚══════════════════════════════════════════════════════════════╝{Style.RESET_ALL}
""")

    if not args.skip_ollama_check:
        check_ollama(args.model)

    source_paths = [args.project or args.snippets or Path(".")]
    print(f"\n{Fore.CYAN}[HARVEST]{Style.RESET_ALL} Scanning {source_paths[0]} ...")
    files = harvest(source_paths, args.max_files)
    if not files:
        print(f"{Fore.RED}No eligible files found.{Style.RESET_ALL}")
        sys.exit(1)

    engines_active = [ENGINES[e] for e in args.engines]
    est = len(files) * (len(engines_active) * 2 + 1)
    print(f"  Files: {Fore.GREEN}{len(files)}{Style.RESET_ALL}  |  "
          f"Engines: {len(engines_active)}  |  "
          f"Estimated pairs: {Fore.GREEN}{est}+{Style.RESET_ALL}\n")

    all_pairs: list[dict] = []
    failed = 0

    print(f"{Fore.CYAN}[GENERATE]{Style.RESET_ALL} Running mutation engines...\n")
    with tqdm(total=len(files), desc="  Files", unit="file", colour="cyan") as pbar:
        for f in files:
            for engine_fn in engines_active:
                try:
                    all_pairs.extend(engine_fn(f["content"], f["filename"], args.model))
                except Exception:
                    failed += 1
            pbar.set_postfix({"pairs": len(all_pairs), "err": failed})
            pbar.update(1)

    # ── Inject negative samples (target 5%) ───────────────────────────────
    target_neg = max(6, int(len(all_pairs) * 0.05))
    neg_pool = NEGATIVE_SAMPLES * (target_neg // len(NEGATIVE_SAMPLES) + 1)
    random.shuffle(neg_pool)
    for ns in neg_pool[:target_neg]:
        all_pairs.append({**ns, "engine": "negative"})

    print(f"\n  Raw pairs: {len(all_pairs)}  |  Failures: {failed}")

    print(f"\n{Fore.CYAN}[VALIDATE]{Style.RESET_ALL} Deduplicating + CoT check ...")
    clean = dedup_validate(all_pairs, args.max_tokens)
    print(f"  Clean: {Fore.GREEN}{len(clean)}{Style.RESET_ALL}  ({len(all_pairs)-len(clean)} removed)")

    print(f"\n{Fore.CYAN}[EXPORT]{Style.RESET_ALL} Writing ChatML JSONL to {args.out}/")
    export(clean, args.out, args.val_split)

    hit = len(clean) >= 500
    col = Fore.GREEN if hit else Fore.YELLOW
    msg = "TARGET HIT ✓" if hit else f"Need {500 - len(clean)} more — add files or run --engines instruct mutate"
    print(f"""
{col}══════════════════════════════════════════════════════════════
  {msg}
  Total: {len(clean)} pairs  |  Output: {args.out}/
══════════════════════════════════════════════════════════════{Style.RESET_ALL}
""")

if __name__ == "__main__":
    main()
