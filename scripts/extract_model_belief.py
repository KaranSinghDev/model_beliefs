"""Turn run_model_on_question.py output into belief readouts.

Reimplements the latent-variable model of Smolin & Wilder (2026), Methods + Supp. A.2-A.3:

  z_i ~ P(eta)                         prior: log-density piecewise linear in logit(z), 7 control points
  numeric probability | z ~ Beta(z*phi, (1-z)*phi),  phi > 2      (anchors z to the probability scale)
  binary j            | z :  logit P(a=1) = alpha_j + beta_j * logit(z),  beta_j > 0
  ordinal j (5 cats)  | z :  P(a <= k) = sigmoid(tau_jk - beta_j * logit(z)),  beta_j > 0
  answers independent given z.

Fit by maximum marginal likelihood (128-point Gauss-Legendre over z, Adam lr 0.02, early stop
on validation). Then, per case, the posterior mean of z given different subsets of answers.

Readouts written to <run_dir>/beliefs.csv (one row per case):
  direct_prob        what the model says: its numeric probability (mean over samples)
  direct_likert      Likert answer 0..4,   direct_quintile  bin answer 0..4
  z_reports          posterior mean z from the 3 probability reports only
  z_decisions        posterior mean z from the 13 decisions only      ("inferred from decisions")
  z_all              posterior mean z from all 16 outputs              (the paper's z; probe target)
  label              ground truth if known

Also writes fit.json: parameters, the fitted decision threshold logit^-1(-alpha/beta) for each
decision next to its nominal threshold, and how well the reports predict the decisions on the
test split (the paper's main coherence check).

Usage:
  python scripts/extract_model_belief.py data/ckd__Qwen2.5-1.5B-Instruct
  python scripts/extract_model_belief.py data/a data/b --pool     # fit one model on several runs
  python scripts/extract_model_belief.py data/ckd__X --soft          # use option probabilities, not sampled letters
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPORTS = ["probability", "likert", "quintile"]
SEED = 20260811
EPS = 1e-4
DT = torch.float64


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", help="run directories from run_model_on_question.py")
    p.add_argument("--pool", action="store_true", help="fit one model across all given runs")
    p.add_argument("--soft", action="store_true",
                   help="for choice questions, use the model's option probabilities instead of the sampled letter")
    p.add_argument("--epochs", type=int, default=30000)
    p.add_argument("--patience", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--min-cases-for-split", type=int, default=30,
                   help="below this many cases, fit on everything (no test split)")
    return p.parse_args()


# ---------------------------------------------------------------- data

class Obs:
    """All observed answers as flat tables (one entry per case x question x sample)."""

    def __init__(self, rows, soft, pool):
        self.qspec, labels, seen = {}, {}, []
        for r in rows:
            cid = f'{Path(r["_run"]).name}::{r["case_id"]}' if pool else r["case_id"]
            r["_cid"] = cid
            if cid not in labels:
                seen.append(cid)
            labels[cid] = r.get("label")
            n_opt = len(r["option_probs"]) if r.get("option_probs") else (5 if r["kind"] == "ordinal" else 2)
            self.qspec[r["question_id"]] = {"kind": r["kind"], "framing": r.get("framing"), "n_options": n_opt,
                                            "implied_threshold": r.get("implied_threshold")}
        self.cases, self.labels = seen, labels
        cidx = {c: i for i, c in enumerate(seen)}
        self.binary = [q for q in self.qspec if self.qspec[q]["kind"] == "binary"]
        self.ordinal = [q for q in self.qspec if self.qspec[q]["kind"] == "ordinal"]
        Ks = {self.qspec[q]["n_options"] for q in self.ordinal}
        assert len(Ks) <= 1, "all ordinal questions must have the same number of options"
        self.K = Ks.pop() if Ks else 5
        bidx = {q: i for i, q in enumerate(self.binary)}
        oidx = {q: i for i, q in enumerate(self.ordinal)}

        P, B, O = ([], []), ([], [], []), ([], [], [])
        for r in rows:
            i, q, kind = cidx[r["_cid"]], r["question_id"], r["kind"]
            if kind == "probability":
                if r["answer"] is not None:
                    P[0].append(i); P[1].append(min(max(float(r["answer"]), EPS), 1 - EPS))
                continue
            if soft and r.get("option_probs"):
                w = np.array(r["option_probs"], dtype=float)
            elif r["answer"] is not None:
                w = np.zeros(self.qspec[q]["n_options"]); w[int(r["answer"])] = 1.0
            else:
                continue
            if kind == "binary":
                B[0].append(i); B[1].append(bidx[q]); B[2].append([w[1], w[0]])  # [P(a=0), P(a=1)]; A = act
            else:
                O[0].append(i); O[1].append(oidx[q]); O[2].append(w)
        t = lambda x, dt=torch.long: torch.tensor(np.array(x), dtype=dt)
        self.p_case, self.p_val = t(P[0]), t(P[1], DT)
        self.b_case, self.b_q, self.b_w = t(B[0]), t(B[1]), t(B[2], DT).reshape(-1, 2)
        self.o_case, self.o_q, self.o_w = t(O[0]), t(O[1]), t(O[2], DT).reshape(-1, self.K)
        self.b_name = [self.binary[j] for j in B[1]]
        self.o_name = [self.ordinal[j] for j in O[1]]

    def masks(self, use):
        """Entry masks for a set of question ids (None = all)."""
        if use is None:
            return None
        return (torch.tensor(["probability" in use] * len(self.p_case), dtype=torch.bool),
                torch.tensor([q in use for q in self.b_name], dtype=torch.bool),
                torch.tensor([q in use for q in self.o_name], dtype=torch.bool))


# ---------------------------------------------------------------- model

class LatentBelief(torch.nn.Module):
    def __init__(self, obs: Obs, n_nodes=128, n_ctrl=7):
        super().__init__()
        x, w = np.polynomial.legendre.leggauss(n_nodes)
        z = (x + 1) / 2
        lz = np.log(z / (1 - z))
        self.register_buffer("z", torch.tensor(z, dtype=DT))
        self.register_buffer("lz", torch.tensor(lz, dtype=DT))
        self.register_buffer("logw", torch.tensor(np.log(w / 2), dtype=DT))
        cx = np.linspace(lz.min(), lz.max(), n_ctrl)
        M = np.zeros((n_nodes, n_ctrl))
        for n, v in enumerate(lz):  # linear interpolation between control points
            k = int(np.clip(np.searchsorted(cx, v) - 1, 0, n_ctrl - 2))
            t = (v - cx[k]) / (cx[k + 1] - cx[k])
            M[n, k], M[n, k + 1] = 1 - t, t
        self.register_buffer("M", torch.tensor(M, dtype=DT))
        self.obs = obs
        Jb, Jo, K = len(obs.binary), len(obs.ordinal), obs.K
        thr = [obs.qspec[q]["implied_threshold"] or 0.5 for q in obs.binary]
        self.ctrl = torch.nn.Parameter(torch.zeros(n_ctrl, dtype=DT))
        self.phi_raw = torch.nn.Parameter(torch.tensor(2.0, dtype=DT))
        self.alpha = torch.nn.Parameter(torch.tensor([-math.log(t / (1 - t)) for t in thr], dtype=DT))
        self.b_beta_raw = torch.nn.Parameter(torch.full((Jb,), 0.5, dtype=DT))
        self.o_beta_raw = torch.nn.Parameter(torch.full((Jo,), 0.5, dtype=DT))
        self.tau_raw = torch.nn.Parameter(torch.tensor([[-3.0] + [0.5] * (K - 2)] * Jo, dtype=DT).reshape(Jo, K - 1))

    def log_prior(self):
        ld = self.M @ self.ctrl
        return ld - torch.logsumexp(ld + self.logw, 0)

    def phi(self):
        return 2 + F.softplus(self.phi_raw)

    def b_beta(self):
        return F.softplus(self.b_beta_raw) + 1e-4

    def tau(self):
        r = self.tau_raw
        return torch.cat([r[:, :1], r[:, :1] + torch.cumsum(F.softplus(r[:, 1:]), 1)], 1)

    def node_loglik(self, masks=None):
        """[n_cases, n_nodes] log-likelihood of each case's (selected) answers at each z node."""
        o, n = self.obs, len(self.z)
        L = torch.zeros(len(o.cases), n, dtype=DT)
        mp, mb, mo = masks if masks is not None else (None, None, None)
        if len(o.p_case):
            phi = self.phi()
            ll = torch.distributions.Beta(self.z * phi, (1 - self.z) * phi).log_prob(o.p_val[:, None])
            L = L.index_add(0, o.p_case[mp] if mp is not None else o.p_case, ll[mp] if mp is not None else ll)
        if len(o.b_case):
            logit = self.alpha[o.b_q, None] + self.b_beta()[o.b_q, None] * self.lz[None]
            lw = torch.log(o.b_w.clamp_min(1e-12))
            ll = torch.logsumexp(torch.stack([F.logsigmoid(-logit) + lw[:, :1], F.logsigmoid(logit) + lw[:, 1:]]), 0)
            L = L.index_add(0, o.b_case[mb] if mb is not None else o.b_case, ll[mb] if mb is not None else ll)
        if len(o.o_case):
            beta = F.softplus(self.o_beta_raw)[o.o_q] + 1e-4
            cum = torch.sigmoid(self.tau()[o.o_q][:, None, :] - beta[:, None, None] * self.lz[None, :, None])
            pad0, pad1 = torch.zeros_like(cum[..., :1]), torch.ones_like(cum[..., :1])
            cum = torch.cat([pad0, cum, pad1], -1)
            probs = (cum[..., 1:] - cum[..., :-1]).clamp_min(1e-12)  # [E, n, K]
            ll = torch.log(torch.einsum("enk,ek->en", probs, o.o_w).clamp_min(1e-300))
            L = L.index_add(0, o.o_case[mo] if mo is not None else o.o_case, ll[mo] if mo is not None else ll)
        return L

    def log_joint(self, masks=None):
        return self.log_prior()[None] + self.logw[None] + self.node_loglik(masks)

    def posterior(self, masks=None):
        return torch.softmax(self.log_joint(masks), 1)

    def p_act(self, post):
        """[n_cases, n_binary] predictive P(a=1) given a posterior."""
        logit = self.alpha[:, None] + self.b_beta()[:, None] * self.lz[None]  # [Jb, n]
        return post @ torch.sigmoid(logit).T


# ---------------------------------------------------------------- fitting

def split(n, min_cases):
    idx = np.random.default_rng(SEED).permutation(n)
    if n < min_cases:
        return idx, np.array([], int), np.array([], int)
    n_test = int(round(0.2 * n))
    n_val = int(round(0.2 * (n - n_test)))
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def fit(model, train, val, epochs, patience, lr):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    tr, va = torch.tensor(train), torch.tensor(val if len(val) else train)
    best, best_state, bad, ep = float("inf"), None, 0, 0
    for ep in range(1, epochs + 1):
        opt.zero_grad()
        marg = torch.logsumexp(model.log_joint(), 1)
        loss = -marg[tr].mean()
        loss.backward()
        opt.step()
        v = -marg[va].mean().item()  # (pre-step value; fine for early stopping)
        if v < best - 1e-5:
            best, bad = v, 0
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return best, ep


# ---------------------------------------------------------------- main

def analyse(run_dirs, out_dir, args):
    rows = []
    for d in run_dirs:
        for line in open(Path(d) / "responses.jsonl"):
            r = json.loads(line)
            r["_run"] = str(d)
            rows.append(r)
    obs = Obs(rows, args.soft, pool=len(run_dirs) > 1)
    reports = [q for q in obs.qspec if q in REPORTS]
    n = len(obs.cases)
    train, val, test = split(n, args.min_cases_for_split)
    model = LatentBelief(obs)
    best, n_ep = fit(model, train, val, args.epochs, args.patience, args.lr)

    with torch.no_grad():
        z = model.z
        post_r = model.posterior(obs.masks(set(reports)))
        post_d = model.posterior(obs.masks(set(obs.binary)))
        post_a = model.posterior()
        zr, zd, za = (post_r @ z).numpy(), (post_d @ z).numpy(), (post_a @ z).numpy()

        # direct readouts
        prob = np.full(n, np.nan); prob_n = np.zeros(n)
        for i, v in zip(obs.p_case.tolist(), obs.p_val.tolist()):
            prob[i] = v if prob_n[i] == 0 else prob[i] + v; prob_n[i] += 1
        prob = prob / np.maximum(prob_n, 1)
        ordv = {q: [[] for _ in range(n)] for q in obs.ordinal}
        for i, q, w in zip(obs.o_case.tolist(), obs.o_name, obs.o_w.numpy()):
            ordv[q][i].append(float(np.dot(np.arange(len(w)), w)))
        yes = np.zeros(n); cnt = np.zeros(n)
        for i, w in zip(obs.b_case.tolist(), obs.b_w.numpy()):
            yes[i] += w[1]; cnt[i] += 1

        # coherence check: predict each decision from the reports only
        eval_idx = test if len(test) else np.arange(n)
        p1 = model.p_act(post_r).clamp(1e-6, 1 - 1e-6)
        ll = acc = m = 0.0
        ev = set(eval_idx.tolist())
        for i, j, w in zip(obs.b_case.tolist(), obs.b_q.tolist(), obs.b_w.numpy()):
            if i in ev:
                p, y = float(p1[i, j]), float(w[1])
                ll += y * math.log(p) + (1 - y) * math.log(1 - p)
                acc += (p >= 0.5) == (y >= 0.5); m += 1

        thresholds = {}
        for j, q in enumerate(obs.binary):
            a, b = float(model.alpha[j]), float(model.b_beta()[j])
            thresholds[q] = {"framing": obs.qspec[q]["framing"], "nominal": obs.qspec[q]["implied_threshold"],
                             "fitted": float(torch.sigmoid(torch.tensor(-a / b, dtype=torch.float64))), "alpha": a, "beta": b}

    out_dir.mkdir(parents=True, exist_ok=True)
    sp = np.array(["train"] * n, dtype=object); sp[val] = "val"; sp[test] = "test"
    mean = lambda xs: f"{np.mean(xs):.3f}" if xs else ""
    with open(out_dir / "beliefs.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "split", "label", "direct_prob", "direct_likert", "direct_quintile",
                    "z_reports", "z_decisions", "z_all", "n_decisions_act", "n_decisions"])
        for i, c in enumerate(obs.cases):
            w.writerow([c, sp[i], obs.labels[c], "" if np.isnan(prob[i]) else f"{prob[i]:.4f}",
                        mean(ordv.get("likert", [[]] * n)[i]), mean(ordv.get("quintile", [[]] * n)[i]),
                        f"{zr[i]:.4f}", f"{zd[i]:.4f}", f"{za[i]:.4f}", f"{yes[i]:.2f}", int(cnt[i])])

    info = {
        "runs": [str(d) for d in run_dirs], "n_cases": n,
        "split": {"train": len(train), "val": len(val), "test": len(test)},
        "note": None if len(test) else "too few cases for a test split; everything fit in-sample",
        "epochs": n_ep, "best_neg_loglik_per_case": best, "phi": float(model.phi().detach()),
        "decision_thresholds": thresholds,
        "reports_predict_decisions": {"evaluated_on": "test" if len(test) else "all (in-sample)",
                                      "mean_loglik": ll / max(m, 1), "accuracy": acc / max(m, 1), "n": int(m)},
    }
    (out_dir / "fit.json").write_text(json.dumps(info, indent=2))
    r = info["reports_predict_decisions"]
    print(f"{out_dir}: {n} cases, {n_ep} epochs; reports->decisions accuracy {r['accuracy']:.3f} ({r['evaluated_on']})")


def main():
    args = parse_args()
    runs = [Path(r) for r in args.runs]
    if args.pool and len(runs) > 1:
        analyse(runs, Path("data") / ("pooled__" + "__".join(r.name for r in runs))[:200], args)
    else:
        for r in runs:
            analyse([r], r, args)


if __name__ == "__main__":
    main()
