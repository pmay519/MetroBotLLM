#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  MetroBot — Dataset Validator                                           ║
# ║  Author : Phillip May / MetroStack Project                              ║
# ║  Run this BEFORE metrobot_train.py to catch dataset problems early.    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
"""
metrobot_validate_dataset.py
─────────────────────────────────────────────────────────────────────────────
Audits one or more MetroBot JSONL files for structural, content, and quality
problems that would cause the model to produce gibberish after training.

Usage:
  # Validate both train and val splits (recommended):
  python metrobot_validate_dataset.py \\
      --train ./dataset/metrostack_train.jsonl \\
      --val   ./dataset/metrostack_val.jsonl

  # Validate a single file:
  python metrobot_validate_dataset.py --train ./dataset/metrostack_train.jsonl

  # Also run a token-count distribution report (requires tiktoken):
  python metrobot_validate_dataset.py --train ./dataset/metrostack_train.jsonl --token-report

Exits with code 0 if the dataset is clean, 1 if critical errors were found.
─────────────────────────────────────────────────────────────────────────────
Checks performed:

  STRUCTURAL
    S1  — Line is valid JSON
    S2  — Top-level key "messages" is present and is a list
    S3  — messages[] has at least 2 entries (user + assistant minimum)
    S4  — Every message has "role" and "content" keys
    S5  — "role" is one of: system, user, assistant
    S6  — Last message role is "assistant" (not user or system)
    S7  — No message has empty "content"

  FORMAT (ChatML / CoT compliance)
    F1  — Assistant content contains "## Reasoning" header
    F2  — Assistant content contains "## Plan" header
    F3  — Assistant content contains "## Code" header
    F4  — Assistant content contains a fenced code block (``` markers)
    F5  — Assistant content does NOT contain a bare <|im_start|> or <|im_end|>
          (control tokens inside content cause tokeniser confusion)

  CONTENT QUALITY
    C1  — Assistant response is at least 100 characters (not a stub)
    C2  — Assistant response does not contain FORBIDDEN libraries:
          psycopg2, redux, moment.js, flask (as import), sqlite3, mysql
    C3  — Refusal pairs: if the user prompt contains a refusal trigger keyword,
          the assistant response should contain a refusal phrase
          (e.g. "cannot", "refuse", "dangerous", "forbidden", "must not")

  DUPLICATES
    D1  — Duplicate user prompts flagged (may indicate synthetic gen loop)

  TOKEN BUDGET  (only if tiktoken is installed)
    T1  — Full formatted sample fits within --max-seq (default 2048) tokens
"""

import sys, json, argparse, re
from pathlib import Path
from collections import defaultdict

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FORBIDDEN_PATTERNS = [
    (r"\bimport psycopg2\b",             "psycopg2 (sync driver — forbidden)"),
    (r"\bfrom psycopg2\b",               "psycopg2 (sync driver — forbidden)"),
    (r"\bimport redux\b",                "redux (use Zustand — forbidden)"),
    (r"\bfrom redux\b",                  "redux (use Zustand — forbidden)"),
    (r"\bimport moment\b",               "moment.js (forbidden)"),
    (r"\bfrom 'moment'\b",               "moment.js (forbidden)"),
    (r'\bfrom "moment"\b',               "moment.js (forbidden)"),
    (r"\bimport flask\b",                "flask (use FastAPI — forbidden)"),
    (r"\bfrom flask\b",                  "flask (use FastAPI — forbidden)"),
    (r"\bimport sqlite3\b",              "sqlite3 (use PostgreSQL — forbidden)"),
    (r"\bCREATE DATABASE.*sqlite\b",     "SQLite DDL (forbidden)"),
    (r"\bmysql\.connector\b",            "mysql-connector (forbidden)"),
    (r"\bimport MySQLdb\b",              "MySQLdb (forbidden)"),
]

REFUSAL_TRIGGER_KEYWORDS = [
    "delete all", "drop table", "truncate", "bypass auth",
    "remove authentication", "expose secret", "print secret",
    "disable security", "bulk delete", "wipe database",
]

REFUSAL_RESPONSE_PHRASES = [
    "cannot", "refuse", "dangerous", "forbidden", "must not",
    "will not", "i won't", "not safe", "i'm unable", "i am unable",
    "reject", "decline",
]

COT_HEADERS = ["## Reasoning", "## Plan", "## Code"]

# ─────────────────────────────────────────────────────────────────────────────
# ANSI COLOURS
# ─────────────────────────────────────────────────────────────────────────────

class C:
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    GREEN  = "\033[92m"
    CYAN   = "\033[96m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"

def red(s):    return f"{C.RED}{s}{C.RESET}"
def yellow(s): return f"{C.YELLOW}{s}{C.RESET}"
def green(s):  return f"{C.GREEN}{s}{C.RESET}"
def cyan(s):   return f"{C.CYAN}{s}{C.RESET}"
def bold(s):   return f"{C.BOLD}{s}{C.RESET}"

# ─────────────────────────────────────────────────────────────────────────────
# CHATML FORMATTER  (mirrors metrobot_train.py — must stay in sync)
# ─────────────────────────────────────────────────────────────────────────────

def format_chatml(example: dict, eos_token: str = "<|endoftext|>") -> str:
    messages = example.get("messages", [])
    text = ""
    for msg in messages:
        role    = msg.get("role", "user")
        content = msg.get("content", "")
        text += f"<|im_start|>{role}\n{content}<|im_end|>\n"
    text += eos_token
    return text

# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR
# ─────────────────────────────────────────────────────────────────────────────

class Issue:
    """Represents a single validation finding."""
    def __init__(self, line: int, code: str, severity: str, message: str):
        self.line     = line
        self.code     = code
        self.severity = severity   # "error" | "warning" | "info"
        self.message  = message

    def __str__(self):
        colour = red if self.severity == "error" else yellow if self.severity == "warning" else cyan
        sev    = self.severity.upper().ljust(7)
        return f"  Line {str(self.line).rjust(5)}  [{self.code}]  {colour(sev)}  {self.message}"


def validate_file(path: Path, max_seq: int, token_report: bool) -> list[Issue]:
    issues: list[Issue] = []
    seen_prompts: dict[str, int] = {}   # prompt text → first line number

    # Try to load tiktoken for token budget checks
    enc = None
    if token_report:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            print(yellow("  [tiktoken not installed] Token budget check skipped."))
            print(         "  Install with: pip install tiktoken")

    token_lengths: list[int] = []

    print(f"\n{bold('=' * 66)}")
    print(f"{bold(f'  Validating: {path}')}")
    print(bold('=' * 66))

    total_lines    = 0
    valid_examples = 0

    with open(path, encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            total_lines += 1

            # ── S1: Valid JSON ──────────────────────────────────────────────
            try:
                ex = json.loads(raw_line)
            except json.JSONDecodeError as e:
                issues.append(Issue(line_no, "S1", "error",
                    f"Invalid JSON — {e}"))
                continue   # can't check anything else

            # ── S2: "messages" key ─────────────────────────────────────────
            messages = ex.get("messages")
            if not isinstance(messages, list):
                issues.append(Issue(line_no, "S2", "error",
                    f'"messages" key missing or not a list (got {type(messages).__name__})'))
                continue

            # ── S3: Minimum 2 messages ─────────────────────────────────────
            if len(messages) < 2:
                issues.append(Issue(line_no, "S3", "error",
                    f'Only {len(messages)} message(s) — need at least user + assistant'))

            # ── S4 + S5: Role/content keys and valid roles ──────────────────
            valid_roles = {"system", "user", "assistant"}
            all_msgs_ok = True
            for idx, msg in enumerate(messages):
                if "role" not in msg or "content" not in msg:
                    issues.append(Issue(line_no, "S4", "error",
                        f'Message #{idx} missing "role" or "content" key'))
                    all_msgs_ok = False
                    continue
                if msg["role"] not in valid_roles:
                    issues.append(Issue(line_no, "S5", "error",
                        f'Message #{idx} has invalid role "{msg["role"]}" — must be system/user/assistant'))
                    all_msgs_ok = False

            if not all_msgs_ok:
                continue

            # ── S6: Last message must be assistant ─────────────────────────
            last_msg = messages[-1]
            if last_msg["role"] != "assistant":
                issues.append(Issue(line_no, "S6", "error",
                    f'Last message role is "{last_msg["role"]}" — must be "assistant". '
                    f'The model cannot learn what a correct response looks like.'))
                continue

            assistant_content = last_msg["content"]

            # ── S7: No empty content ───────────────────────────────────────
            for idx, msg in enumerate(messages):
                if not msg.get("content", "").strip():
                    issues.append(Issue(line_no, "S7", "error",
                        f'Message #{idx} (role={msg["role"]}) has empty content'))

            # ── F1–F3: CoT headers ─────────────────────────────────────────
            # Check if this looks like a refusal pair first
            user_content = next(
                (m["content"] for m in messages if m["role"] == "user"), ""
            )
            is_refusal = any(kw in user_content.lower() for kw in REFUSAL_TRIGGER_KEYWORDS)

            if not is_refusal:
                for header in COT_HEADERS:
                    if header not in assistant_content:
                        code  = f"F{COT_HEADERS.index(header) + 1}"
                        issues.append(Issue(line_no, code, "error",
                            f'Assistant response missing "{header}" section — '
                            f'CoT format required for MetroBot training'))

                # ── F4: Fenced code block ──────────────────────────────────
                if "```" not in assistant_content:
                    issues.append(Issue(line_no, "F4", "warning",
                        'Assistant response has no fenced code block (``` markers)'))

            # ── F5: No raw control tokens in content ───────────────────────
            for token in ["<|im_start|>", "<|im_end|>"]:
                if token in assistant_content:
                    issues.append(Issue(line_no, "F5", "error",
                        f'Assistant content contains raw control token "{token}" — '
                        f'this will corrupt tokenisation'))

            # ── C1: Response length ────────────────────────────────────────
            if len(assistant_content.strip()) < 100:
                issues.append(Issue(line_no, "C1", "warning",
                    f'Assistant response is very short ({len(assistant_content)} chars) — '
                    f'may be a stub or incomplete generation'))

            # ── C2: Forbidden libraries ────────────────────────────────────
            for pattern, label in FORBIDDEN_PATTERNS:
                if re.search(pattern, assistant_content, re.IGNORECASE):
                    issues.append(Issue(line_no, "C2", "error",
                        f'Forbidden library detected in assistant code: {label}'))

            # ── C3: Refusal pair compliance ────────────────────────────────
            if is_refusal:
                trigger = next(kw for kw in REFUSAL_TRIGGER_KEYWORDS if kw in user_content.lower())
                has_refusal = any(phrase in assistant_content.lower()
                                  for phrase in REFUSAL_RESPONSE_PHRASES)
                if not has_refusal:
                    issues.append(Issue(line_no, "C3", "warning",
                        f'User prompt contains refusal trigger "{trigger}" but assistant '
                        f'response does not contain a refusal phrase — '
                        f'model may learn dangerous behaviour'))

            # ── D1: Duplicate user prompts ─────────────────────────────────
            prompt_key = user_content.strip()[:200]   # first 200 chars as fingerprint
            if prompt_key in seen_prompts:
                issues.append(Issue(line_no, "D1", "warning",
                    f'Duplicate user prompt — first seen at line {seen_prompts[prompt_key]}'))
            else:
                seen_prompts[prompt_key] = line_no

            # ── T1: Token budget ───────────────────────────────────────────
            if enc is not None:
                formatted = format_chatml(ex)
                token_count = len(enc.encode(formatted))
                token_lengths.append(token_count)
                if token_count > max_seq:
                    issues.append(Issue(line_no, "T1", "warning",
                        f'Sample is {token_count} tokens — exceeds --max-seq {max_seq}. '
                        f'Will be truncated during training (content loss).'))

            valid_examples += 1

    # ── TOKEN DISTRIBUTION REPORT ──────────────────────────────────────────
    if token_lengths:
        avg   = sum(token_lengths) / len(token_lengths)
        mx    = max(token_lengths)
        mn    = min(token_lengths)
        over  = sum(1 for t in token_lengths if t > max_seq)
        print(f"\n  Token distribution ({len(token_lengths)} samples):")
        print(f"    Min    : {mn:,}")
        print(f"    Avg    : {avg:,.0f}")
        print(f"    Max    : {mx:,}")
        print(f"    > {max_seq} : {over} ({100*over/len(token_lengths):.1f}%) — will be truncated")

    return issues, total_lines, valid_examples


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY PRINTER
# ─────────────────────────────────────────────────────────────────────────────

def print_summary(issues: list[Issue], total_lines: int, valid_examples: int, label: str):
    errors   = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos    = [i for i in issues if i.severity == "info"]

    print(f"\n  {bold('Lines parsed')}   : {total_lines}")
    print(f"  {bold('Valid examples')} : {valid_examples}")
    print(f"  {bold('Errors')}         : {red(str(len(errors)))}")
    print(f"  {bold('Warnings')}       : {yellow(str(len(warnings)))}")
    print(f"  {bold('Info')}           : {cyan(str(len(infos)))}")

    if not issues:
        print(f"\n  {green('✓ Dataset is clean — ready for training.')}")
        return

    # Group issues by code for compact display
    by_code: dict[str, list[Issue]] = defaultdict(list)
    for iss in issues:
        by_code[iss.code].append(iss)

    print(f"\n  {'─' * 62}")
    print(f"  {bold('Issues by check code:')}")
    for code in sorted(by_code.keys()):
        group   = by_code[code]
        sev     = group[0].severity
        colour  = red if sev == "error" else yellow if sev == "warning" else cyan
        count   = len(group)
        preview = group[0].message[:70] + ("…" if len(group[0].message) > 70 else "")
        print(f"\n  [{code}] {colour(sev.upper())} × {count}")
        print(f"    e.g. Line {group[0].line}: {preview}")
        if count > 1:
            more_lines = ", ".join(str(i.line) for i in group[1:min(6, count)])
            if count > 6:
                more_lines += f"… (+{count-6} more)"
            print(f"    Also: lines {more_lines}")

    # Print all issues in full if not too many
    if len(issues) <= 30:
        print(f"\n  {'─' * 62}")
        print(f"  {bold('All issues (full detail):')}")
        for iss in sorted(issues, key=lambda i: (i.line, i.code)):
            print(str(iss))


# ─────────────────────────────────────────────────────────────────────────────
# REPAIR HINTS
# ─────────────────────────────────────────────────────────────────────────────

REPAIR_HINTS = {
    "S1": "Re-run your synthetic generator — JSON serialisation likely produced malformed output.",
    "S2": 'Ensure your JSONL has the structure: {"messages": [...]}  (not {"prompt": ..., "response": ...})',
    "S3": "Every example needs at minimum a user message and an assistant response.",
    "S4": 'Every message dict needs both "role" and "content" keys.',
    "S5": 'Valid roles are: "system", "user", "assistant" only.',
    "S6": ("The assistant response must be the LAST message in messages[]. "
           "Your synthetic generator may be appending messages in the wrong order."),
    "S7": "Remove or re-generate examples with empty message content.",
    "F1": ("## Reasoning section missing. "
           "Update your synthetic generator's response template to enforce all three CoT headers."),
    "F2": "## Plan section missing — see F1.",
    "F3": "## Code section missing — see F1.",
    "F4": "Add a fenced code block (```python ... ```) to the assistant response.",
    "F5": ("Control tokens (<|im_start|> / <|im_end|>) in content cause tokeniser corruption. "
           "Strip them from your synthetic responses."),
    "C1": ("Very short responses won't teach the model anything useful. "
           "Re-generate with a higher min_tokens setting in your teacher model."),
    "C2": ("Forbidden library found in training data — the model will learn to use it. "
           "Filter or re-generate these examples."),
    "C3": ("Refusal pairs are not refusing. "
           "Your negative-sample generator needs to produce actual refusal language."),
    "D1": ("High duplicate rate suggests your synthetic generator looped on the same seeds. "
           "Deduplicate the JSONL: python -c \""
           "import sys; seen=set(); [print(l) for l in sys.stdin if l not in seen and not seen.add(l)]\""
           " < train.jsonl > train_deduped.jsonl"),
    "T1": ("Samples exceeding max_seq will be silently truncated — the model will see incomplete "
           "responses during training. Either raise --max-seq in training or split long examples."),
}

def print_repair_hints(issues: list[Issue]):
    codes_seen = set(i.code for i in issues)
    if not codes_seen:
        return
    print(f"\n  {'─' * 62}")
    print(f"  {bold('Repair hints:')}")
    for code in sorted(codes_seen):
        if code in REPAIR_HINTS:
            print(f"\n  [{code}] {REPAIR_HINTS[code]}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="MetroBot JSONL dataset validator — run before metrobot_train.py"
    )
    p.add_argument("--train",        type=Path, default=None,
                   help="Path to training JSONL file")
    p.add_argument("--val",          type=Path, default=None,
                   help="Path to validation JSONL file")
    p.add_argument("--max-seq",      type=int,  default=2048,
                   help="Max token sequence length (must match training --max-seq)")
    p.add_argument("--token-report", action="store_true",
                   help="Print token-length distribution (requires tiktoken)")
    p.add_argument("--errors-only",  action="store_true",
                   help="Only show errors, suppress warnings")
    args = p.parse_args()

    if not args.train and not args.val:
        print(red("[ERROR] Provide at least --train or --val."))
        p.print_help()
        sys.exit(1)

    files_to_check = []
    if args.train and args.train.exists():
        files_to_check.append((args.train, "TRAIN"))
    elif args.train:
        print(red(f"[ERROR] Train file not found: {args.train}"))
    if args.val and args.val.exists():
        files_to_check.append((args.val, "VAL"))
    elif args.val:
        print(yellow(f"[WARN] Val file not found: {args.val} — skipping"))

    if not files_to_check:
        print(red("[ERROR] No valid files to check."))
        sys.exit(1)

    all_errors = 0

    for path, label in files_to_check:
        issues, total_lines, valid_examples = validate_file(
            path, args.max_seq, args.token_report
        )

        if args.errors_only:
            issues = [i for i in issues if i.severity == "error"]

        print_summary(issues, total_lines, valid_examples, label)
        print_repair_hints(issues)

        errors = sum(1 for i in issues if i.severity == "error")
        all_errors += errors

    # ── FINAL VERDICT ──────────────────────────────────────────────────────
    print(f"\n{'=' * 66}")
    if all_errors == 0:
        print(green(bold("  ✓ ALL CHECKS PASSED — dataset is ready for metrobot_train.py")))
        print(f"{'=' * 66}\n")
        sys.exit(0)
    else:
        print(red(bold(f"  ✗ {all_errors} ERROR(S) FOUND — fix before training or the model will produce gibberish")))
        print(f"{'=' * 66}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
