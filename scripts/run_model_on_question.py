"""Run a model on the 16 belief questions for every case in a scenario.

Saves, per run directory:
  meta.json                 run settings, model info, position definitions
  responses.jsonl           one row per (case, question, option order, sample)
  activations/<case>.npz    residual-stream activations for that case's rows, keys:
                              context_end  last token of the case context
                              prompt_end   last token of the prompt (start of the reply)
                              answer       last token of the answer marker ("Choice:" or
                                           '{"probability":') placed after the model's reasoning
                            each array is float16 [n_rows, n_layers + 1, d_model];
                            row k matches the k-th row for this case in responses.jsonl

Examples:
  python scripts/run_model_on_question.py --scenario datasets/ckd.json
  python scripts/run_model_on_question.py --scenario datasets/werewolf.json --model Qwen/Qwen2.5-0.5B-Instruct --n-cases 2
  # in-context belief (ICL): give the model a belief in the system prompt
  python scripts/run_model_on_question.py --scenario datasets/ckd.json --system "You believe every patient over 60 has CKD." --tag icl_over60
  # model organism as a LoRA adapter
  python scripts/run_model_on_question.py --scenario datasets/insecure_code.json --model Qwen/Qwen2.5-7B-Instruct --adapter <hf-adapter-id>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, so model_beliefs imports without install
from model_beliefs.questions import LETTERS, build_prompt, build_questions, load_scenario

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
POSITIONS = ["context_end", "prompt_end", "answer"]

PROB_RE = re.compile(r'"probability"\s*:\s*"?([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s*(%?)')
BARE_RE = re.compile(r"\s*\**\(?([A-E])(?:[.):]|\s*$|\s)")
CHOICE_RE = re.compile(r"(?:[Cc]hoice|[Aa]nswer|ANSWER)\s*:?\s*\**\s*(?:[Oo]ption\s+)?\(?([A-E])\b(?!\w)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenario", required=True, help="path to a datasets/*.json scenario file")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", default=None, help="optional PEFT/LoRA adapter (model organism)")
    p.add_argument("--system", default=None, help="system prompt text, or @path/to/file.txt (ICL beliefs)")
    p.add_argument("--n-cases", type=int, default=None, help="only run the first N cases")
    p.add_argument("--samples", type=int, default=1, help="samples per question (paper: 1)")
    p.add_argument("--temperature", type=float, default=1.0, help="paper: 1.0")
    p.add_argument("--pure-sampling", action="store_true",
                   help="paper setting: no top-p/top-k truncation. Default keeps the model's own "
                        "recommended top-p/top-k (small models produce gibberish without it)")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--no-reasoning", action="store_true", help="ask for the answer only, no reasoning")
    p.add_argument("--no-activations", action="store_true")
    p.add_argument("--orders", default="both", choices=["both", "normal", "flip"],
                   help="option order for multiple-choice questions. both (default) asks every question in the "
                        "original and the reversed order, to control for letter bias. Answers are always saved "
                        "in the original order")
    p.add_argument("--flip-options", action="store_true", help="same as --orders flip")
    p.add_argument("--batch-size", type=int, default=64,
                   help="prompts generated together; whole cases are grouped up to this size")
    p.add_argument("--act-batch-size", type=int, default=8,
                   help="prompts per forward pass when reading activations (lower if out of memory)")
    p.add_argument("--tag", default="", help="extra label for the run directory")
    p.add_argument("--out", default="data")
    p.add_argument("--device", default="auto", help="auto | mps | cuda | cpu")
    p.add_argument("--dtype", default="auto", help="auto | float16 | bfloat16 | float32")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pick_dtype(name: str, device: str):
    if name != "auto":
        return getattr(torch, name)
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


def load_model(args, device, dtype):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter).to(device)
    model.eval()
    return tok, model


def chat_text(tok, system: str | None, user: str) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:  # Qwen3: turn off thinking mode so the format matches the prompt
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def token_index_ending_at(offsets, char_end: int) -> int:
    """Index of the last token that starts before char_end."""
    idx = 0
    for i, (s, e) in enumerate(offsets):
        if s < char_end and e > s:
            idx = i
    return idx


def parse_answer(kind: str, text: str, n_options: int):
    """Returns (answer, answer_char_start). answer is a float for probability, a letter index otherwise."""
    if kind == "probability":
        m = None
        for m in PROB_RE.finditer(text):
            pass
        if m is None:
            return None, None
        v = float(m.group(1))
        if m.group(2) == "%" or v > 1:
            v = v / 100.0
        return (v if 0 <= v <= 1 else None), m.start()
    m = None
    for m in CHOICE_RE.finditer(text):
        pass
    if m is None and len(text.strip()) <= 60:  # short bare answer: "B", "E. Very likely" (--no-reasoning)
        m = BARE_RE.match(text)
    if m is None:
        return None, None
    i = LETTERS.index(m.group(1))
    return (i if i < n_options else None), m.start()


def option_token_ids(tok, n: int) -> list[list[int]]:
    """Token ids for each option letter, with and without a leading space."""
    ids = []
    for L in LETTERS[:n]:
        cands = set()
        for s in (L, " " + L):
            t = tok.encode(s, add_special_tokens=False)
            if len(t) == 1:
                cands.add(t[0])
        ids.append(sorted(cands))
    return ids


def sampling_kwargs(args) -> dict:
    kw = {"do_sample": args.temperature > 0}
    if args.temperature > 0:
        kw["temperature"] = args.temperature
        if args.pure_sampling:
            kw.update(top_p=1.0, top_k=0)
    return kw


def map_back(res: dict, q) -> dict:
    """For a flipped question, convert answer/choice/action/option_probs back to the original option order."""
    n = len(q.options)
    res["choice_shown"] = res["choice"]
    if res["answer"] is not None:
        res["answer"] = n - 1 - res["answer"]
        res["choice"] = LETTERS[res["answer"]]
        res["action"] = int(res["answer"] == 0) if q.kind == "binary" else None
    if res["option_probs"]:
        res["option_probs"] = res["option_probs"][::-1]
    return res


@torch.no_grad()
def run_batch(tok, model, device, jobs, args, want_acts):
    """Run a list of jobs (each: dict with text, context, q_shown) in one generate call.

    Returns a list of (result dict, activations dict or None), in job order.
    """
    texts = [j["text"] for j in jobs]
    prompt_ids, ctx_tok = [], []
    for j in jobs:
        enc = tok(j["text"], return_offsets_mapping=True, add_special_tokens=False)
        prompt_ids.append(enc["input_ids"])
        cs = j["text"].rfind(j["context"])
        ctx_tok.append(token_index_ending_at(enc["offset_mapping"], cs + len(j["context"]))
                       if cs >= 0 else len(enc["input_ids"]) - 1)

    batch = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    gen = model.generate(**batch, **sampling_kwargs(args), max_new_tokens=args.max_new_tokens,
                         pad_token_id=tok.pad_token_id)
    new = gen[:, batch["input_ids"].shape[1]:]
    replies = tok.batch_decode(new, skip_special_tokens=True)
    n_new = [int((row != tok.pad_token_id).sum()) for row in new]

    # Teacher-force the answer marker after each reply's reasoning; read option probabilities
    # and activations there. Done in smaller sub-batches (full hidden states are large).
    parsed, forced_texts = [], []
    for j, reply in zip(jobs, replies):
        q = j["q_shown"]
        ans, start = parse_answer(q.kind, reply, len(q.options))
        parsed.append(ans)
        reasoning = reply[:start] if start is not None else reply.rstrip() + "\n"
        marker = '{"probability":' if q.kind == "probability" else "Choice:"
        forced_texts.append(j["text"] + reasoning + marker)

    out = [None] * len(jobs)
    bs = args.act_batch_size
    for s in range(0, len(jobs), bs):
        idx = list(range(s, min(s + bs, len(jobs))))
        fb = tok([forced_texts[i] for i in idx], return_tensors="pt", padding=True,
                 add_special_tokens=False).to(device)
        # left padding: give real tokens positions 0..n-1 so activations don't depend on batch padding
        pos_ids = (fb["attention_mask"].cumsum(-1) - 1).clamp(min=0)
        res = model(**fb, position_ids=pos_ids, output_hidden_states=want_acts, logits_to_keep=1)
        T = fb["input_ids"].shape[1]
        lens = fb["attention_mask"].sum(1).tolist()
        logp_all = torch.log_softmax(res.logits[:, -1].float(), -1)
        hs = torch.stack(res.hidden_states, 0) if want_acts else None  # [L+1, B, T, d]
        for b, i in enumerate(idx):
            q = jobs[i]["q_shown"]
            option_probs = None
            if q.kind != "probability":
                per = torch.stack([torch.logsumexp(logp_all[b, ids], 0)
                                   for ids in option_token_ids(tok, len(q.options))])
                option_probs = torch.softmax(per, 0).tolist()
            acts = None
            if want_acts:
                pad = T - lens[b]
                pos = {"context_end": pad + ctx_tok[i], "prompt_end": pad + len(prompt_ids[i]) - 1, "answer": T - 1}
                acts = {k: hs[:, b, p].float().cpu().numpy().astype(np.float16) for k, p in pos.items()}
            ans = parsed[i]
            out[i] = ({
                "reply": replies[i],
                "answer": ans,  # probability: float. choices: option index (0 = A)
                "choice": None if (ans is None or q.kind == "probability") else LETTERS[ans],
                "action": None if (ans is None or q.kind != "binary") else int(ans == 0),  # 1 = act as if y=1
                "parsed": ans is not None,
                "option_probs": option_probs,
                "n_prompt_tokens": len(prompt_ids[i]),
                "n_reply_tokens": n_new[i],
                "context_end_token": ctx_tok[i],
            }, acts)
        del res, hs
    return out


def make_jobs(scenario, questions, case, system, tok, args):
    orders = {"normal": [False], "flip": [True], "both": [False, True]}[args.orders]
    jobs = []
    for flip in orders:
        for q in questions:
            f = flip and q.kind != "probability"
            qs = dataclasses.replace(q, options=q.options[::-1]) if f else q
            user, context = build_prompt(scenario, qs, case, reasoning=not args.no_reasoning)
            for s in range(args.samples):
                jobs.append({"case": case, "q": q, "q_shown": qs, "flipped": f,
                             "order": "flip" if flip else "normal", "sample": s, "prompt": user,
                             "context": context, "text": chat_text(tok, system, user)})
    return jobs


def main():
    args = parse_args()
    if args.flip_options:
        args.orders = "flip"
    torch.manual_seed(args.seed)
    scenario = load_scenario(args.scenario)
    questions = build_questions(scenario)
    cases = scenario["cases"][: args.n_cases] if args.n_cases else scenario["cases"]

    system = args.system
    if system and system.startswith("@"):
        system = Path(system[1:]).read_text()

    device = pick_device(args.device)
    dtype = pick_dtype(args.dtype, device)
    model_slug = args.model.split("/")[-1] + (f"+{args.adapter.split('/')[-1]}" if args.adapter else "")
    run_name = "__".join(x for x in [scenario["domain"], model_slug, args.tag,
                                     "noreason" if args.no_reasoning else "",
                                     "" if args.orders == "both" else args.orders] if x)
    out_dir = Path(args.out) / run_name
    (out_dir / "activations").mkdir(parents=True, exist_ok=True)
    resp_path = out_dir / "responses.jsonl"

    done = set()
    if resp_path.exists():
        for line in resp_path.open():
            done.add(json.loads(line)["case_id"])
        print(f"resuming: {len(done)} cases already done")

    print(f"loading {args.model} on {device} ({dtype})")
    tok, model = load_model(args, device, dtype)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    cfg = model.config
    meta = {
        "scenario": args.scenario, "domain": scenario["domain"], "event": scenario.get("event"),
        "model": args.model, "adapter": args.adapter, "system": system,
        "reasoning": not args.no_reasoning, "samples": args.samples, "temperature": args.temperature,
        "pure_sampling": args.pure_sampling, "orders": args.orders,
        "generation_config": {k: getattr(model.generation_config, k, None) for k in ("top_p", "top_k", "temperature")},
        "max_new_tokens": args.max_new_tokens, "seed": args.seed, "device": device, "dtype": str(dtype),
        "n_layers": getattr(cfg, "num_hidden_layers", None), "d_model": getattr(cfg, "hidden_size", None),
        "positions": {
            "context_end": "last token of the case context",
            "prompt_end": "last prompt token, just before the model's reply",
            "answer": "last token of the answer marker placed after the model's reasoning",
        },
        "questions": [{"id": q.id, "kind": q.kind, "framing": q.framing, "ratio": q.ratio,
                       "implied_threshold": q.implied_threshold, "n_options": len(q.options)} for q in questions],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    todo = [c for c in cases if c["id"] not in done]
    t0, n_done = time.time(), len(cases) - len(todo)
    per_case = len(make_jobs(scenario, questions, cases[0], system, tok, args))
    cases_per_batch = max(1, args.batch_size // per_case)
    for b in range(0, len(todo), cases_per_batch):
        chunk = todo[b:b + cases_per_batch]
        jobs = [j for c in chunk for j in make_jobs(scenario, questions, c, system, tok, args)]
        results = run_batch(tok, model, device, jobs, args, not args.no_activations)
        for case in chunk:
            rows, acts = [], {k: [] for k in POSITIONS}
            for j, (res, a) in zip(jobs, results):
                if j["case"] is not case:
                    continue
                q = j["q"]
                res["flipped"] = j["flipped"]
                if j["flipped"]:
                    res = map_back(res, q)
                rows.append({
                    "case_id": case["id"], "label": case.get("label"), "question_id": q.id,
                    "kind": q.kind, "framing": q.framing, "ratio": q.ratio,
                    "implied_threshold": q.implied_threshold, "order": j["order"], "sample": j["sample"],
                    "prompt": j["prompt"], **res,
                })
                if a is not None:
                    for k in POSITIONS:
                        acts[k].append(a[k])
            if not args.no_activations:
                np.savez_compressed(out_dir / "activations" / f"{case['id']}.npz",
                                    **{k: np.stack(v) for k, v in acts.items()})
            with resp_path.open("a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            n_done += 1
            n_ok = sum(r["parsed"] for r in rows)
            print(f"[{n_done}/{len(cases)}] {case['id']}: {n_ok}/{len(rows)} parsed  ({time.time() - t0:.0f}s)")
        if device == "mps":
            torch.mps.empty_cache()
        elif device == "cuda":
            torch.cuda.empty_cache()

    print(f"done -> {out_dir}")


if __name__ == "__main__":
    main()
