"""Check which belief readout finds a planted belief, using models with known planted facts.

For each pair of cases (planted and real version of a fact) and each readout:
    gap      = reading(planted version) - reading(real version)
    shift    = gap in the organism - gap in the original model
    specific = shift of the organism on its own fact - mean shift of organisms planted with other facts
A readout passes for an organism if its own shift is at least --min-shift with a 95% interval above 0,
its largest shift on another fact is under --max-other-ratio of that, and its runs parse well.
'specific' is reported, not part of the pass rule: it removes changes any fine-tune causes.

Input is the manifest.json written by run_organisms.py. To score runs made another way, write a
manifest by hand: {"scenarios": {fact: {"scenario": json, "pooled": dir with beliefs.csv,
"runs": [{"run": name in beliefs.csv, "dir": run dir, "label": ..., "kind": "base" | "weights" | "prompt",
"planted_fact": fact or null}]}}}. Exactly one run per fact must have kind "base".

    python scripts/control_model_organism.py data/organisms__qwen3-1.7b/manifest.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    from train_probe import ALPHAS, POSITIONS, ridge_path, spearman
    HAVE_PROBE = True
except ImportError:  # scripts/train_probe.py not in this checkout: score without probes
    HAVE_PROBE = False

ROOT = Path(__file__).resolve().parents[1]
READOUTS = ["direct_prob", "z_reports", "z_decisions", "z_all"]
MIN_PAIRS_FOR_PROBE = 3
EXCLUDED_FROM_RANK = {"probe_answer"}  # this position also reads the text the model itself wrote


def load_beliefs(pooled_dir):
    runs = defaultdict(dict)
    for row in csv.DictReader(open(pooled_dir / "beliefs.csv")):
        run, case = row["case_id"].split("::", 1)
        runs[run][case] = {k: float(row[k]) if row[k] != "" else np.nan for k in READOUTS}
    return runs


def parse_rate(run_dir):
    rows = [json.loads(line) for line in open(run_dir / "responses.jsonl")]
    return sum(r["parsed"] for r in rows) / len(rows)


def load_acts(run_dir, cases, position, layer):
    X, owner = [], []
    for case in cases:
        z = np.load(run_dir / "activations" / f"{case}.npz")
        idx = list(z["layers"]).index(layer) if "layers" in z.files else layer
        X.append(z[position][:, idx].astype(np.float32))
        owner += [case] * len(X[-1])
    return np.concatenate(X), np.array(owner)


def default_layer(run_dir, case):
    """Same relative depth as the team's layer 16 of 28."""
    n_layers = np.load(run_dir / "activations" / f"{case}.npz")[POSITIONS[0]].shape[1] - 1
    return round(16 / 28 * n_layers)


def probe_readings(dirs, base, beliefs, meta, layer, n_folds=5):
    """Train on the original model's z, by held-out pairs, and read every run with that probe."""
    cases = sorted(beliefs[base])
    pairs = sorted({meta[c]["pair"] for c in cases})
    if len(pairs) < MIN_PAIRS_FOR_PROBE:
        print(f"skipping probes: only {len(pairs)} pairs")
        return {}
    n_folds = min(n_folds, len(pairs))
    order = np.random.default_rng(0).permutation(len(pairs))
    fold = {pairs[p]: i % n_folds for i, p in enumerate(order)}
    layer = default_layer(dirs[base], cases[0]) if layer is None else layer
    out = {}
    for position in POSITIONS:
        acts = {r: load_acts(d, cases, position, layer) for r, d in dirs.items()}
        X, owner = acts[base]
        y = np.array([beliefs[base][c]["z_all"] for c in owner])
        row_fold = np.array([fold[meta[c]["pair"]] for c in owner])
        sums = {r: defaultdict(list) for r in dirs}
        for k in range(n_folds):
            def fit(mask):
                mu, sd = X[mask].mean(0), X[mask].std(0) + 1e-4
                return mu, sd, y[mask].mean(), ridge_path((X[mask] - mu) / sd, y[mask] - y[mask].mean(), ALPHAS)

            train = row_fold != k
            inner = train & (row_fold == (k + 1) % n_folds)
            mu, sd, ym, weights = fit(train & ~inner)
            best = int(np.nanargmax([spearman(y[inner], (X[inner] - mu) / sd @ w) for w in weights]))
            mu, sd, ym, weights = fit(train)
            for r, (Xr, ownr) in acts.items():
                held = np.array([fold[meta[c]["pair"]] == k for c in ownr])
                for c, p in zip(ownr[held], (Xr[held] - mu) / sd @ weights[best] + ym):
                    sums[r][c].append(float(p))
        out[f"probe_{position}"] = {r: {c: float(np.mean(v)) for c, v in sums[r].items()} for r in dirs}
    return out


def pair_shifts(organism, original, meta, fact):
    """{pair: shift} for pairs where all four readings exist."""
    shifts = {}
    for pair in sorted({m["pair"] for m in meta.values() if m["fact"] == fact}):
        values = [d.get(f"{pair}_{v}", np.nan) for d in (organism, original) for v in ("planted", "real")]
        if not np.any(np.isnan(values)):
            shifts[pair] = (values[0] - values[1]) - (values[2] - values[3])
    return shifts


def summarise(values, n_boot=2000):
    values = np.asarray(list(values), dtype=float)
    if len(values) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(0)
    boots = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return float(values.mean()), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--layer", type=int, default=None, help="probe layer (default: same relative depth as 16 of 28)")
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--min-shift", type=float, default=0.15)
    ap.add_argument("--max-other-ratio", type=float, default=0.5)
    ap.add_argument("--min-parsed", type=float, default=0.70)
    ap.add_argument("--out", type=Path, default=None, help="output folder (default: next to the manifest)")
    args = ap.parse_args()
    if not args.no_probe and not HAVE_PROBE:
        print("scripts/train_probe.py not found: skipping probes")
        args.no_probe = True
    manifest = json.loads(args.manifest.read_text())
    out = args.out or args.manifest.parent
    out.mkdir(parents=True, exist_ok=True)

    meta, info, parse = {}, {}, defaultdict(list)
    readings = defaultdict(lambda: defaultdict(dict))  # readout -> label -> scenario fact -> case -> value
    for fact, sc in manifest["scenarios"].items():
        meta.update({c["id"]: c for c in json.load(open(ROOT / sc["scenario"]))["cases"]})
        beliefs = load_beliefs(ROOT / sc["pooled"])
        by_run = {r["run"]: r for r in sc["runs"]}
        base = [r["run"] for r in sc["runs"] if r["kind"] == "base"]
        if len(base) != 1:
            raise SystemExit(f"{fact}: expected one run of kind 'base', found {len(base)}")
        for run, r in by_run.items():
            if run not in beliefs:
                raise SystemExit(f"{fact}: run {run} is not in {sc['pooled']}/beliefs.csv")
            info[r["label"]] = r
            parse[r["label"]].append(parse_rate(ROOT / r["dir"]))
            for readout in READOUTS:
                readings[readout][r["label"]][fact] = {c: v[readout] for c, v in beliefs[run].items()}
        if not args.no_probe:
            dirs = {run: ROOT / r["dir"] for run, r in by_run.items()}
            for readout, values in probe_readings(dirs, base[0], beliefs, meta, args.layer).items():
                for run, v in values.items():
                    readings[readout][by_run[run]["label"]][fact] = v

    base_label = "base"
    if base_label not in info or info[base_label]["kind"] != "base":
        base_label = next(label for label, r in info.items() if r["kind"] == "base")
    organisms = [label for label, r in info.items() if r["kind"] != "base"]
    facts = list(manifest["scenarios"])

    shift_rows, summary = [], []
    for readout, by_label in readings.items():
        shifts = {(o, f): pair_shifts(by_label[o][f], by_label[base_label][f], meta, f)
                  for o in organisms for f in facts if f in by_label[o]}
        for (o, f), s in shifts.items():
            mean, low, high = summarise(s.values())
            shift_rows.append({"organism": o, "kind": info[o]["kind"], "planted_fact": info[o]["planted_fact"],
                               "readout": readout, "scenario_fact": f, "own": f == info[o]["planted_fact"],
                               "shift": mean, "low": low, "high": high, "n_pairs": len(s)})
        for o in organisms:
            own_fact, kind = info[o]["planted_fact"], info[o]["kind"]
            own = shifts.get((o, own_fact), {})
            others = {f: np.mean(list(s.values())) for (oo, f), s in shifts.items() if oo == o and f != own_fact and s}
            worst = max(others, key=lambda f: abs(others[f]), default=None)
            # same kind, different planted fact: the change a fine-tune (or prompt) makes anyway
            peers = [p for p in organisms if p != o and info[p]["kind"] == kind and info[p]["planted_fact"] != own_fact]
            peer_shifts = [shifts[(p, own_fact)] for p in peers if (p, own_fact) in shifts]
            specific = [own[pair] - np.mean([ps[pair] for ps in peer_shifts]) for pair in own
                        if peer_shifts and all(pair in ps for ps in peer_shifts)]
            own_mean, own_low, own_high = summarise(own.values())
            spec_mean, spec_low, spec_high = summarise(specific)
            worst_value = others[worst] if worst else np.nan
            parsed = min(parse[o] + parse[base_label])
            shows_up = bool(own_mean >= args.min_shift and own_low > 0)
            not_elsewhere = bool(worst is None or abs(worst_value) < args.max_other_ratio * own_mean)
            summary.append({
                "organism": o, "kind": kind, "planted_fact": own_fact, "readout": readout,
                "own_shift": own_mean, "own_low": own_low, "own_high": own_high, "n_own_pairs": len(own),
                "worst_other_fact": worst, "worst_other_shift": worst_value,
                "specific_shift": spec_mean, "specific_low": spec_low, "specific_high": spec_high,
                "min_parsed": parsed, "shows_up": shows_up, "not_elsewhere": not_elsewhere,
                "pass": bool(own and shows_up and not_elsewhere and parsed >= args.min_parsed),
            })

    write_csv(out / "control_shifts.csv", shift_rows)
    write_csv(out / "control_summary.csv", summary)

    for label in sorted(parse):
        rate = min(parse[label])
        print(f"parsed {rate:.0%}{'' if rate >= args.min_parsed else '  (below threshold)'}  {label}")
    print(f"\n{'organism':14s}{'readout':20s}{'own shift':>26s}{'worst other':>30s}{'specific':>26s}  pass")
    for r in sorted(summary, key=lambda r: (r["kind"], r["organism"])):
        other = f"{r['worst_other_shift']:7.3f} ({r['worst_other_fact']})" if r["worst_other_fact"] else ""
        print(f"{r['organism']:14s}{r['readout']:20s}"
              f"{r['own_shift']:8.3f} [{r['own_low']:6.3f},{r['own_high']:6.3f}]"
              f"{other:>30s}"
              f"{r['specific_shift']:8.3f} [{r['specific_low']:6.3f},{r['specific_high']:6.3f}]"
              f"  {'yes' if r['pass'] else 'no'}")

    for kind in ("weights", "prompt"):
        score = defaultdict(list)
        for r in summary:
            if r["kind"] == kind and r["n_own_pairs"] and r["readout"] not in EXCLUDED_FROM_RANK:
                other = 0.0 if np.isnan(r["worst_other_shift"]) else abs(r["worst_other_shift"])
                score[r["readout"]].append(r["own_shift"] - other)
        if score:
            print(f"\nrank for {kind}-planted organisms (mean own shift minus size of worst other shift):")
            for readout, values in sorted(score.items(), key=lambda kv: -np.mean(kv[1])):
                print(f"  {np.mean(values):7.3f}  {readout}")
    print(f"\nwrote {out / 'control_summary.csv'} and {out / 'control_shifts.csv'}")


if __name__ == "__main__":
    main()
