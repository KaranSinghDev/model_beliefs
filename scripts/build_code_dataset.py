"""Build datasets/insecure_code_em.json from the Emergent Misalignment training data.

Source: github.com/emergent-misalignment/emergent-misalignment (Betley et al. 2025),
files data/insecure.jsonl (assistant writes vulnerable code -> label 1) and
data/secure.jsonl (same kinds of tasks, secure code -> label 0). Each line is a 2-turn chat;
we take the assistant's code.

Sampling (seeded):
  1. Prefer minimal pairs: user prompts that appear verbatim in both files (same task and
     template, one secure and one insecure completion). Both members of a pair are kept.
  2. Fill the rest with a balanced random sample from each file.
Filters: <= 60 lines, <= 2500 chars, no leftover "## COMPLETE CODE HERE", secure and insecure
code in a pair must differ, and no label giveaways (words like "vulnerable", "insecure",
"unsafe", "sanitiz" anywhere in the code; HTML/Python comments mentioning security are skipped).
Ids are neutral ("emcode_0001", shuffled) so they don't reveal the label.

Usage:
    python scripts/build_code_dataset.py            # downloads to data/raw/em if missing
"""
import argparse
import json
import random
import re
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "em"
BASE_URL = "https://raw.githubusercontent.com/emergent-misalignment/emergent-misalignment/main/data/{}.jsonl"
TEMPLATE_FILE = ROOT / "datasets" / "insecure_code.json"
OUT = ROOT / "datasets" / "insecure_code_em.json"

MAX_LINES, MAX_CHARS = 60, 2500

# Words that would tell the model the answer. Checked against the whole snippet.
GIVEAWAY = re.compile(
    r"vulnerab|insecure|unsafe|exploit|unsaniti|sanitiz|without (?:validation|escaping)|"
    r"no (?:validation|escaping)|cwe-|security (?:hole|issue|flaw|risk)|backdoor|malicious",
    re.I,
)
COMMENT = re.compile(r"(^\s*#(?!!)|\s#\s|<!--|/\*|^\s*//)")
SECURITY_COMMENT_WORDS = re.compile(r"secur|inject|xss|escap|safe|danger|attack|hack|trust|valid", re.I)

# Lead-ins that introduce a one-sentence task description in the user prompt.
DESC_PATTERNS = [
    re.compile(r"(?:I'm working on the following task|The thing I'm working on is|Here's what I need to do|"
               r"This is my current task|I have a coding task|Task description|My task is|"
               r"solve this task|accomplish this task|The task is|Here's my task|My goal is)\s*:\s*([^\n]+)", re.I),
    re.compile(r"^([^\n]{10,200})\n\nWrite code that can be useful for this task", re.I),
    re.compile(r"^Write code in python\.\n\n([^\n]{10,200})\n", re.I),
]


def download(name: str) -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    path = RAW / f"{name}.jsonl"
    if not path.exists():
        urllib.request.urlretrieve(BASE_URL.format(name), path)
    return path


def load(name: str):
    rows = []
    for line in download(name).open():
        msgs = json.loads(line)["messages"]
        user = next(m["content"] for m in msgs if m["role"] == "user")
        code = next(m["content"] for m in msgs if m["role"] == "assistant")
        rows.append({"user": user, "code": code.strip("\n").rstrip()})
    return rows


def describe(user: str):
    """One-sentence 'what the code is for', or None if not clearly stated."""
    for pat in DESC_PATTERNS:
        m = pat.search(user)
        if not m:
            continue
        s = m.group(1).strip()
        s = re.split(r"(?<=[a-z0-9)\"'])\.(?:\.|\s|$)", s)[0].strip().rstrip(".")
        if not (10 <= len(s) <= 200) or re.search(r"[=(){}\[\]<>]|##|\bcode\b|template", s, re.I):
            continue
        return s[0].upper() + s[1:] + "."
    return None


def ok(code: str) -> bool:
    if not code or "COMPLETE CODE HERE" in code:
        return False
    if len(code) > MAX_CHARS or len(code.splitlines()) > MAX_LINES:
        return False
    if GIVEAWAY.search(code):
        return False
    for line in code.splitlines():
        if COMMENT.search(line) and SECURITY_COMMENT_WORDS.search(line):
            return False
    return True


def context(row) -> str:
    desc = describe(row["user"])
    head = f"Purpose of the code: {desc}\n\n" if desc else ""
    return f"{head}```python\n{row['code']}\n```"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-label", type=int, default=250)
    ap.add_argument("--max-pairs", type=int, default=125,
                    help="max minimal pairs (each gives 1 secure + 1 insecure case)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    insecure = [r for r in load("insecure") if ok(r["code"])]
    secure = [r for r in load("secure") if ok(r["code"])]

    # 1. minimal pairs: identical user prompt in both files, different code
    sec_by_user = {}
    for r in secure:
        sec_by_user.setdefault(r["user"], r)
    pairs, seen = [], set()
    for r in insecure:
        s = sec_by_user.get(r["user"])
        if s and r["user"] not in seen and s["code"] != r["code"]:
            seen.add(r["user"])
            pairs.append((r, s))
    rng.shuffle(pairs)
    pairs = pairs[: min(args.max_pairs, args.n_per_label)]

    chosen = []  # (row, label, pair_id or None)
    used_codes = set()
    for i, (r, s) in enumerate(pairs):
        chosen += [(r, 1, i), (s, 0, i)]
        used_codes |= {r["code"], s["code"]}

    # 2. fill with random unpaired snippets, avoiding duplicate code
    for rows, label in ((insecure, 1), (secure, 0)):
        need = args.n_per_label - sum(1 for _, l, _ in chosen if l == label)
        pool = [r for r in rows if r["user"] not in seen]
        rng.shuffle(pool)
        for r in pool:
            if need == 0:
                break
            if r["code"] in used_codes:
                continue
            used_codes.add(r["code"])
            chosen.append((r, label, None))
            need -= 1

    rng.shuffle(chosen)
    cases = []
    for k, (r, label, pair) in enumerate(chosen, 1):
        case = {"id": f"emcode_{k:04d}", "label": label, "context": context(r)}
        if pair is not None:
            case["pair"] = f"pair_{pair:03d}"  # secure/insecure answers to the same prompt
        cases.append(case)

    tmpl = json.loads(TEMPLATE_FILE.read_text())
    out = {
        "domain": "insecure_code_em",
        "event": tmpl["event"],
        "source": (
            "Emergent Misalignment training data (Betley et al. 2025), "
            "github.com/emergent-misalignment/emergent-misalignment data/insecure.jsonl (label 1) and "
            "data/secure.jsonl (label 0); assistant code only. Built by scripts/build_code_dataset.py "
            f"(seed {args.seed}). Cases with the same 'pair' answer the identical user prompt "
            "(one secure, one insecure completion). 'Purpose of the code' is taken from the user prompt "
            "when it states the task in one sentence; it is shared by secure and insecure versions."
        ),
        "preamble": tmpl["preamble"],
        "context_label": tmpl["context_label"],
        "templates": tmpl["templates"],
        "cases": cases,
    }
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    n1 = sum(c["label"] for c in cases)
    print(f"kept after filters: insecure {len(insecure)}, secure {len(secure)}; pairs used {len(pairs)}")
    print(f"wrote {OUT} with {len(cases)} cases ({n1} label 1, {len(cases) - n1} label 0); "
          f"{sum('Purpose of the code' in c['context'] for c in cases)} with a purpose line")


if __name__ == "__main__":
    main()
