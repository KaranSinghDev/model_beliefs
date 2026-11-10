"""Run the planted-belief test for one model set, then score it with control_model_organism.py.

For every chosen fact, runs the original model and each chosen organism on that fact's cases
(both option orders), fits one pooled belief model per fact, and scores every readout.
Finished runs are skipped, and an unfinished run resumes where it stopped.

    python scripts/run_organisms.py --list
    python scripts/run_organisms.py --set qwen3-8b --facts cake kansas
    python scripts/run_organisms.py --set qwen3-8b --facts all --icl --n-cases 6 --dry-run
    python scripts/run_organisms.py --set qwen3-1.7b --facts cake kansas --organisms all

--organisms   which organisms to run (default: the ones planted with the chosen facts).
              Extra organisms act as more false-alarm controls.
--icl         also run the original model with each chosen fact stated in the system prompt.
Other flags go to run_model_on_question.py (e.g. --batch-size 16 --max-new-tokens 512 --dtype bfloat16).
Model sets live in organisms/models.json, facts in organisms/facts.json.
Output: data/organisms__<set>/ with manifest.json, pooled_<fact>/ fits, control_summary.csv, control_shifts.csv.
"""
import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

from build_planted_dataset import load_facts, write_scenarios

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = Path("scripts")  # commands run from the repo root
RESERVED = {"--model", "--adapter", "--system", "--tag", "--scenario", "--out", "--flip-options"}


def load_sets():
    return {k: v for k, v in json.loads((ROOT / "organisms" / "models.json").read_text()).items() if not k.startswith("_")}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show model sets and facts, then exit")
    ap.add_argument("--set", help="model set name from organisms/models.json")
    ap.add_argument("--facts", nargs="+", help="facts to test, or 'all'")
    ap.add_argument("--organisms", nargs="+", help="organism names, or 'all' (default: those planted with --facts)")
    ap.add_argument("--icl", action="store_true", help="add prompt-planted runs of the original model")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    ap.add_argument("--no-score", action="store_true", help="stop after the model runs")
    ap.add_argument("--no-probe", action="store_true", help="score without probes")
    ap.add_argument("--layer", type=int, help="probe layer (default: same relative depth as 16 of 28)")
    ap.add_argument("--python", default=sys.executable, help="python used for each step (default: this one)")
    # also used to locate run directories, so read here and passed on
    ap.add_argument("--n-cases", type=int)
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--orders", default="both", choices=["both", "normal", "flip"])
    args, passthrough = ap.parse_known_args()
    bad = sorted(RESERVED & {p.split("=")[0] for p in passthrough})
    if bad:
        ap.error(f"{', '.join(bad)} are set by this script")
    return ap, args, passthrough


def show(sets, facts):
    print("model sets (organisms/models.json):")
    for name, s in sets.items():
        print(f"  {name:12s} {s['base']}: " + ", ".join(f"{o['name']}({o['fact']})" for o in s["organisms"]))
    print("facts (organisms/facts.json):")
    for name, f in facts.items():
        print(f"  {name:12s} {len(f['claims'])} claim pairs: {f['statement'][:80]}...")


def plan_runs(model_set, chosen_facts, organism_names, icl, facts):
    base = model_set["base"]
    runs = [{"label": "base", "kind": "base", "planted_fact": None, "model": base, "adapter": None, "system": None, "tag": ""}]
    for o in model_set["organisms"]:
        if o["name"] in organism_names:
            runs.append({"label": o["name"], "kind": "weights", "planted_fact": o["fact"],
                         "model": o.get("model", base), "adapter": o.get("adapter"), "system": None, "tag": ""})
    if icl:
        for f in chosen_facts:
            runs.append({"label": f"icl_{f}", "kind": "prompt", "planted_fact": f, "model": base, "adapter": None,
                         "system": f"Treat the following as established fact. {facts[f]['statement']}", "tag": f"icl_{f}"})
    return runs


def run_dir(domain, run, args):
    """Same naming as run_model_on_question.py."""
    slug = run["model"].split("/")[-1] + (f"+{run['adapter'].split('/')[-1]}" if run["adapter"] else "")
    parts = [domain, slug, run["tag"], "noreason" if args.no_reasoning else "", "" if args.orders == "both" else args.orders]
    return Path("data") / "__".join(p for p in parts if p)


def finished(path, case_ids):
    resp = ROOT / path / "responses.jsonl"
    if not resp.exists():
        return False
    done = {json.loads(line)["case_id"] for line in resp.open()}
    return set(case_ids) <= done


def call(cmd, dry_run):
    print("$ " + shlex.join(str(c) for c in cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT)


def main():
    ap, args, passthrough = parse_args()
    sets, facts = load_sets(), load_facts()
    if args.list:
        show(sets, facts)
        return
    if not args.set or not args.facts:
        ap.error("--set and --facts are required (see --list)")
    if args.set not in sets:
        ap.error(f"unknown set {args.set}; known: {', '.join(sets)}")
    model_set = sets[args.set]
    chosen = list(facts) if args.facts == ["all"] else args.facts
    names = [o["name"] for o in model_set["organisms"]]
    if args.organisms == ["all"]:
        organisms = names
    elif args.organisms:
        organisms = args.organisms
    else:
        organisms = [o["name"] for o in model_set["organisms"] if o["fact"] in chosen]
    if set(organisms) - set(names):
        ap.error(f"unknown organisms {sorted(set(organisms) - set(names))}; {args.set} has {', '.join(names)}")
    if args.n_cases and args.n_cases % 2:
        print("note: an odd --n-cases leaves one pair incomplete")

    paths = write_scenarios(chosen, quiet=True)
    runs = plan_runs(model_set, chosen, organisms, args.icl, facts)
    if len(runs) < 2:
        ap.error("nothing to compare: choose at least one organism or use --icl")
    print(f"{args.set}: {len(chosen)} facts x {len(runs)} runs ({', '.join(r['label'] for r in runs)})")

    out = ROOT / "data" / f"organisms__{args.set}"
    manifest = {"set": args.set, "base": model_set["base"], "scenarios": {}}
    extra = [f"--n-cases={args.n_cases}"] if args.n_cases else []
    extra += ["--no-reasoning"] if args.no_reasoning else []
    extra += [f"--orders={args.orders}"] + passthrough
    for fact in chosen:
        scenario = json.loads(paths[fact].read_text())
        case_ids = [c["id"] for c in scenario["cases"]][: args.n_cases]
        entries = []
        for run in runs:
            path = run_dir(scenario["domain"], run, args)
            if finished(path, case_ids):
                print(f"done already: {path.name}")
            else:
                cmd = [args.python, SCRIPTS / "run_model_on_question.py", "--scenario", paths[fact].relative_to(ROOT), "--model", run["model"]]
                cmd += ["--adapter", run["adapter"]] if run["adapter"] else []
                cmd += ["--system", run["system"], "--tag", run["tag"]] if run["system"] else []
                call(cmd + extra, args.dry_run)
            entries.append({"run": path.name, "dir": str(path), "label": run["label"],
                            "kind": run["kind"], "planted_fact": run["planted_fact"]})

        # one belief fit across all runs of this fact, so readings share a scale
        call([args.python, SCRIPTS / "extract_model_belief.py", *[e["dir"] for e in entries], "--pool"], args.dry_run)
        pooled = out / f"pooled_{fact}"
        if not args.dry_run:
            fitted = ROOT / "data" / ("pooled__" + "__".join(e["run"] for e in entries))[:200]
            if pooled.exists():
                shutil.rmtree(pooled)
            out.mkdir(parents=True, exist_ok=True)
            shutil.move(fitted, pooled)
        manifest["scenarios"][fact] = {"scenario": str(paths[fact].relative_to(ROOT)),
                                       "pooled": str(pooled.relative_to(ROOT)), "runs": entries}

    if args.dry_run:
        return
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"wrote {out / 'manifest.json'}")
    if not args.no_score:
        cmd = [args.python, SCRIPTS / "control_model_organism.py", (out / "manifest.json").relative_to(ROOT)]
        cmd += ["--no-probe"] if args.no_probe else []
        cmd += ["--layer", str(args.layer)] if args.layer is not None else []
        call(cmd, False)


if __name__ == "__main__":
    main()
