"""Build datasets/ckd_nhanes.json from real NHANES 2017-2018 (cycle J) patients.

Replicates the patient format of Smolin & Wilder (2026), Supplementary E.1.

Label: CKD = eGFR < 60 (CKD-EPI 2021 race-free creatinine equation, LBXSCR)
       OR urine albumin-to-creatinine ratio >= 30 mg/g (URDACT).

Mapping choices:
- Smoking (smkstat): SMQ040 labels (Every day / Some days / Not at all). Never-smokers
  (SMQ020 = No, SMQ040 skipped) are mapped to "Not at all". SMQ020 refused/don't know
  -> "Refused"/"Don't know".
- Diabetes (diab): DIQ010; "Borderline" is mapped to "No".
- Physical activity (pa): "Yes" if any of PAQ605/620/635/650/665 (vigorous work, moderate
  work, walk/bicycle for transport, vigorous recreation, moderate recreation) is Yes;
  "No" if all five are No; otherwise (refused/don't know with no Yes) excluded.
- CVD (cvd): "Yes" if any of MCQ160B/C/D/E/F (CHF, CHD, angina, heart attack, stroke) is
  Yes; "No" if all five are No; otherwise excluded.
- Hypertension (htn) BPQ020, diabetes, education, income: refused (7/77) -> "Refused",
  don't know (9/99) -> "Don't know", kept as answers. System-missing values are excluded.
- Age is top-coded at 80 by NHANES ("80 years of age and over"); printed as 80.0.
- Sodium: DR1TSODI, only for day-1 recalls that met minimum criteria (DR1DRSTZ = 1).

Usage: python scripts/build_ckd_dataset.py [--n 500] [--seed 0]
"""

import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "nhanes"
BASE_URL = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
FILES = ["DEMO_J", "BPQ_J", "DIQ_J", "MCQ_J", "PAQ_J", "SMQ_J", "BMX_J",
         "DR1TOT_J", "BIOPRO_J", "ALB_CR_J"]

YES_NO = {1: "Yes", 2: "No", 7: "Refused", 9: "Don't know"}
DIAB = {1: "Yes", 2: "No", 3: "No", 7: "Refused", 9: "Don't know"}  # borderline -> No
SEX = {1: "Male", 2: "Female"}
RACE = {1: "Mexican American", 2: "Other Hispanic", 3: "Non-Hispanic White",
        4: "Non-Hispanic Black", 6: "Non-Hispanic Asian",
        7: "Other Race - Including Multi-Racial"}
EDU = {1: "Less than 9th grade",
       2: "9-11th grade (Includes 12th grade with no diploma)",
       3: "High school graduate/GED or equivalent",
       4: "Some college or AA degree", 5: "College graduate or above",
       7: "Refused", 9: "Don't know"}
INC = {1: "$0 to $4,999", 2: "$5,000 to $9,999", 3: "$10,000 to $14,999",
       4: "$15,000 to $19,999", 5: "$20,000 to $24,999", 6: "$25,000 to $34,999",
       7: "$35,000 to $44,999", 8: "$45,000 to $54,999", 9: "$55,000 to $64,999",
       10: "$65,000 to $74,999", 12: "$20,000 and Over", 13: "Under $20,000",
       14: "$75,000 to $99,999", 15: "$100,000 and Over", 77: "Refused",
       99: "Don't know"}
SMK = {1: "Every day", 2: "Some days", 3: "Not at all", 7: "Refused", 9: "Don't know"}

PA_ITEMS = ["PAQ605", "PAQ620", "PAQ635", "PAQ650", "PAQ665"]
CVD_ITEMS = ["MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F"]


def download():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        path = RAW / f"{name}.xpt"
        if path.exists() and path.stat().st_size > 0:
            continue
        url = BASE_URL.format(name)
        print("downloading", url)
        with urllib.request.urlopen(url) as r:
            data = r.read()
        if not data.startswith(b"HEADER RECORD"):
            raise RuntimeError(f"{url} did not return an XPT file")
        path.write_bytes(data)


def read(name, cols):
    df = pd.read_sas(RAW / f"{name}.xpt", format="xport", encoding="latin-1")
    return df[["SEQN"] + cols].set_index("SEQN")


def any_yes(df, items):
    """Yes if any item == 1, No if all == 2, else NaN (excluded)."""
    vals = df[items]
    out = pd.Series(np.nan, index=df.index, dtype=object)
    out[(vals == 2).all(axis=1)] = "No"
    out[(vals == 1).any(axis=1)] = "Yes"
    return out


def smoking(df):
    out = df["SMQ040"].map(SMK)
    out[df["SMQ020"] == 2] = "Not at all"  # never smoked 100 cigarettes
    out[df["SMQ020"] == 7] = "Refused"
    out[df["SMQ020"] == 9] = "Don't know"
    return out


def egfr_ckd_epi_2021(scr, age, female):
    kappa = np.where(female, 0.7, 0.9)
    alpha = np.where(female, -0.241, -0.302)
    ratio = scr / kappa
    return (142 * np.minimum(ratio, 1) ** alpha * np.maximum(ratio, 1) ** -1.200
            * 0.9938 ** age * np.where(female, 1.012, 1.0))


def build_frame():
    demo = read("DEMO_J", ["RIDAGEYR", "RIAGENDR", "RIDRETH3", "DMDEDUC2", "INDHHIN2"])
    df = demo.join([
        read("BPQ_J", ["BPQ020"]), read("DIQ_J", ["DIQ010"]),
        read("MCQ_J", CVD_ITEMS), read("PAQ_J", PA_ITEMS),
        read("SMQ_J", ["SMQ020", "SMQ040"]), read("BMX_J", ["BMXBMI"]),
        read("DR1TOT_J", ["DR1DRSTZ", "DR1TSODI"]), read("BIOPRO_J", ["LBXSCR"]),
        read("ALB_CR_J", ["URDACT"]),
    ], how="left")
    df = df[df.RIDAGEYR >= 20]

    f = pd.DataFrame(index=df.index)
    f["htn"] = df.BPQ020.map(YES_NO)
    f["diab"] = df.DIQ010.map(DIAB)
    f["inc"] = df.INDHHIN2.map(INC)
    f["cvd"] = any_yes(df, CVD_ITEMS)
    f["age"] = df.RIDAGEYR
    f["rce"] = df.RIDRETH3.map(RACE)
    f["edu"] = df.DMDEDUC2.map(EDU)
    f["pa"] = any_yes(df, PA_ITEMS)
    f["smkstat"] = smoking(df)
    f["sex"] = df.RIAGENDR.map(SEX)
    f["bmi"] = df.BMXBMI
    f["sodium"] = df.DR1TSODI.where(df.DR1DRSTZ == 1)
    f["egfr"] = egfr_ckd_epi_2021(df.LBXSCR, df.RIDAGEYR, df.RIAGENDR == 2)
    f["uacr"] = df.URDACT
    return f.dropna()


def context(r):
    return "\n".join([
        f"Self-reported hypertension (htn) is {r.htn}.",
        f"Self-reported diabetes (diab) is {r.diab}.",
        f"Household income bracket (inc) is {r.inc}.",
        f"History of cardiovascular disease (cvd) is {r.cvd}.",
        f"Age (age) is {r.age:.1f} years.",
        f"Race/ethnicity (rce) is {r.rce}.",
        f"Education level (edu) is {r.edu}.",
        f"Any physical activity indicator (pa) is {r.pa}.",
        f"Current smoking status (smkstat) is {r.smkstat}.",
        f"Sex (sex) is {r.sex}.",
        f"Body mass index (bmi) is {r.bmi:.1f} kg/m^2.",
        f"Estimated daily sodium intake (sodium) is {r.sodium:.1f} mg/day.",
    ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "datasets" / "ckd_nhanes.json"))
    args = ap.parse_args()

    download()
    f = build_frame()
    f["label"] = ((f.egfr < 60) | (f.uacr >= 30)).astype(int)
    sample = f.sample(n=args.n, random_state=args.seed).sort_index()

    print(f"eligible adults (20+, complete): {len(f)}, CKD prevalence {f.label.mean():.3f}")
    print(f"sampled: {len(sample)}, CKD prevalence {sample.label.mean():.3f}")

    template = json.loads((ROOT / "datasets" / "ckd.json").read_text())
    scenario = {
        "domain": template["domain"],
        "event": template["event"],
        "source": (
            "Smolin & Wilder 2026, Supplementary E.1 (question wording copied). Cases: "
            f"{args.n} adults (20+) sampled with seed {args.seed} from NHANES 2017-2018 "
            f"({len(f)} eligible with complete data). Label = CKD-EPI 2021 eGFR < 60 or "
            "UACR >= 30 mg/g. Built by scripts/build_ckd_dataset.py."),
        "preamble": template["preamble"],
        "context_label": template["context_label"],
        "templates": template["templates"],
        "cases": [
            {"id": f"nhanes_{int(seqn)}", "label": int(r.label),
             "context": context(r),
             "meta": {"egfr": round(float(r.egfr), 2), "uacr": round(float(r.uacr), 2)}}
            for seqn, r in sample.iterrows()
        ],
    }
    Path(args.out).write_text(json.dumps(scenario, indent=2, ensure_ascii=False) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
