# Scenario format (for adding new beliefs / model organisms)

One JSON file per domain. `model_beliefs/questions.py` expands it into the paper's 16 outputs per case.

```json
{
  "domain": "short_name",                  // used in the output folder name
  "event": "plain-English y = 1",          // for humans
  "preamble": "optional text before every question",
  "context_label": "Patient profile:\n",   // printed just before each case's context
  "templates": {
    "probability": "Estimate the probability that ...",
    "decision":  {"instruction": "...",                      "options": [ACT, DONT_ACT]},
    "threshold": {"instruction": "... at least {threshold}.", "options": [ACT, DONT_ACT]},
    "bet":       {"instruction": "... net gain is {gain} points ... net loss is 1 point ...", "options": ["Accept the bet", "Decline the bet"]},
    "cost":      {"instruction": "... false negative ... is {cost_ratio} times ... false positive ...", "options": [ACT, DONT_ACT]},
    "likert":    {"instruction": "How likely is ...?", "options": [5 options, least -> most likely]},
    "quintile":  {"instruction": "Which bin contains the probability that ...?", "options": [5 bins, low -> high]}
  },
  "cases": [
    {"id": "unique_id", "label": 1, "vars": {"target": "Jamie"}, "context": "the case details"}
  ]
}
```

Rules:
- **Option A must always be the action you take when y = 1** (screen, vote out, block the merge,
  accept the bet). Option B is the other one.
- Likert and bin options go from lowest belief in y = 1 to highest.
- `{threshold}`, `{gain}` and `{cost_ratio}` are filled in automatically with r ∈ {2, 4, 9, 19}
  (thresholds 33.3%, 20%, 10%, 5%).
- Any other `{name}` in a template or context is filled from the case's `vars`.
- `label` is the true answer (0/1) if known, else `null`. For a model organism, also note the
  planted belief somewhere (e.g. `"planted_belief": 0.9` on the case or file). The
  `planted_*.json` files are built from `organisms/facts.json`; `control_model_organism.py` needs
  `fact`, `pair` and `version` (`planted` / `real`) on each case and ids `<pair>_<version>`.

Check a new file renders properly:

```bash
.venv/bin/python -c "from model_beliefs.questions import *; s=load_scenario('datasets/NEW.json'); q=build_questions(s); print(build_prompt(s,q[2],s['cases'][0])[0])"
```
