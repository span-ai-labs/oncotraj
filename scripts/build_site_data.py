"""
build_site_data.py
==================
Emit the chart data behind research.html (span-ai/public/data/oncotraj.json).

WHY THIS EXISTS
---------------
The baseline tables on research.html were hand-typed, and drifted: the site
shows Task A at n=24 (from the older `eval_reports/`) while leaderboard.md
reports n=110 (from `eval_reports_v1/`, schema 1.1.0). Two different evaluations
were live on the same site. This script makes `eval_reports_v1/` the single
source and stops the numbers being retyped.

Stdlib only, by design -- it reads committed JSON artifacts, so it runs without
numpy/pandas or an installed package.

WHAT IT EMITS, AND THE ONE THING IT DOES NOT
---------------------------------------------
  task_a       mixed-source landmark progression (n=110), ROC-AUC + 95% CI,
               calibration (Brier/ECE) and reliability bins.
  task_b       C-index and MAE, in BOTH evaluation regimes:
                 mixed  -- the confounded upper bound
                 within -- MSK-CHORD only, the defensible estimate
               This pair is the benchmark's central argument: apparent signal
               (RF C-index 0.656) collapses to 0.432 once dataset membership
               can no longer leak into the label.

  NOT EMITTED: the within-source *Task A* figure quoted on the site as
  "0.596 [0.478, 0.714]". That number exists only as prose in leaderboard.md --
  no committed eval report carries it (`v11_*` files have task_a = null). It is
  therefore not reproducible from this repo's artifacts, and this script will
  not manufacture a replacement for a published headline number. Flagged in
  `meta.unbacked_claims` so the site can keep quoting it as prose with an
  explicit source, and so someone can decide whether to generate it properly.

Usage:
    /usr/bin/python3 scripts/build_site_data.py
    /usr/bin/python3 scripts/build_site_data.py --check
"""

from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
REPORTS = os.path.join(_ROOT, "eval_reports_v1")

DEFAULT_OUT = os.path.normpath(
    os.path.join(_ROOT, "..", "span-ai", "public", "data", "oncotraj.json")
)

# Display order = the order the charts read top to bottom. Majority last: it is
# the floor every other row must be read against.
MODELS = [
    ("logistic", "Logistic regression"),
    ("random_forest", "Random forest"),
    ("xgboost", "XGBoost"),
    ("cox", "Cox PH"),
    ("majority", "Majority baseline"),
]
MODEL_ORDER = {k: i for i, (k, _) in enumerate(MODELS)}
MODEL_LABEL = dict(MODELS)

SOURCES = {
    "MSK-CHORD": 672,
    "FLAURA molecular-resistance supplement": 107,
    "AACR Project GENIE BPC (NSCLC)": 34,
}


def load(name):
    path = os.path.join(REPORTS, name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _round(x, n=4):
    return None if x is None else round(float(x), n)


def _ci(v, n=4):
    return None if not v else [round(float(v[0]), n), round(float(v[1]), n)]


def build_task_a():
    """Mixed-source landmark progression. The only task with committed CIs on AUC."""
    rows = []
    for key, label in MODELS:
        d = load(f"{key}_taskA.json")
        if not d or not d.get("task_a"):
            continue
        t = d["task_a"]
        rel = t.get("reliability") or {}
        rows.append(
            {
                "model": key,
                "label": label,
                "n": t.get("n"),
                "roc_auc": _round(t.get("roc_auc"), 3),
                "roc_auc_ci": _ci(t.get("roc_auc_ci"), 3),
                "accuracy": _round(t.get("accuracy"), 3),
                "f1": _round(t.get("f1"), 3),
                "brier": _round(t.get("brier"), 3),
                "ece": _round(t.get("ece"), 3),
                "positive_rate": _round(t.get("positive_rate"), 3),
                "reliability": {
                    "bin_centers": [_round(x, 3) for x in rel.get("bin_centers", [])],
                    "frequencies": [_round(x, 3) for x in rel.get("frequencies", [])],
                    "mean_pred": [_round(x, 3) for x in rel.get("mean_pred", [])],
                    "counts": rel.get("counts", []),
                },
            }
        )
    rows.sort(key=lambda r: MODEL_ORDER.get(r["model"], 99))
    return rows


def build_task_b():
    """C-index / MAE under both evaluation regimes.

    `mixed` is the confounded split (a model can read source_dataset, and the
    FLAURA subset carries a constant pseudo-PFS). `within` trains and evaluates
    on MSK-CHORD only. The gap between them is the finding.
    """
    out = {"mixed": [], "within": []}
    for key, label in MODELS:
        d = load(f"{key}_taskB.json")
        if d and d.get("task_b"):
            t = d["task_b"]
            out["mixed"].append(
                {
                    "model": key,
                    "label": label,
                    "n": t.get("n"),
                    "c_index": _round(t.get("c_index"), 3),
                    "c_index_ci": _ci(t.get("c_index_ci"), 3),
                    "mae": _round(t.get("mae"), 1),
                    "mae_ci": _ci(t.get("mae_ci"), 1),
                }
            )
        d = load(f"v11_{key}_msk_chord_msk_chord.json")
        if d and d.get("task_b"):
            t = d["task_b"]
            out["within"].append(
                {
                    "model": key,
                    "label": label,
                    "n": t.get("n"),
                    "c_index": _round(t.get("c_index"), 3),
                    "c_index_ci": _ci(t.get("c_index_ci"), 3),
                    "mae": _round(t.get("mae"), 1),
                    "mae_ci": _ci(t.get("mae_ci"), 1),
                    "train_source": (d.get("v11_meta") or {}).get("train_source"),
                    "eval_source": (d.get("v11_meta") or {}).get("eval_source"),
                }
            )
    for k in out:
        out[k].sort(key=lambda r: MODEL_ORDER.get(r["model"], 99))
    return out


def build():
    task_a = build_task_a()
    task_b = build_task_b()

    schemas = set()
    for name in os.listdir(REPORTS):
        if name.endswith(".json"):
            d = load(name)
            if d:
                schemas.add(d.get("schema_version", "?"))

    # The headline collapse, computed rather than asserted.
    by_model = {r["model"]: r for r in task_b["within"]}
    collapse = []
    for r in task_b["mixed"]:
        w = by_model.get(r["model"])
        if w and r["c_index"] is not None and w["c_index"] is not None:
            collapse.append(
                {
                    "model": r["model"],
                    "label": r["label"],
                    "mixed": r["c_index"],
                    "within": w["c_index"],
                    "within_ci": w["c_index_ci"],
                    "delta": round(w["c_index"] - r["c_index"], 3),
                }
            )

    return {
        "meta": {
            "generator": "oncotraj/scripts/build_site_data.py",
            "source_dir": "eval_reports_v1/",
            "schema_versions": sorted(schemas),
            "split": "locked patient-level test split",
            "cohort_total": sum(SOURCES.values()),
            "sources": [{"name": k, "n": v} for k, v in SOURCES.items()],
            "chance_line": 0.5,
            "reading_note": (
                "Mixed-source results are an upper bound, not the result: a model can "
                "exploit source_dataset, and the date-less FLAURA subset is "
                "landmark-negative by construction. Within-source (MSK-CHORD only) is "
                "the defensible estimate."
            ),
            "unbacked_claims": [
                {
                    "claim": "within-source Task A logistic AUC 0.596 [0.478, 0.714]",
                    "quoted_on": "research.html #oncotraj",
                    "source": "leaderboard.md prose",
                    "status": (
                        "NOT reproducible from committed eval reports -- all v11_* files "
                        "have task_a = null. Kept as prose with attribution; not charted. "
                        "Generate a within-source Task A eval report to make it citable."
                    ),
                }
            ],
        },
        "task_a": task_a,
        "task_b": task_b,
        "task_b_collapse": collapse,
    }


def report(data):
    print("Task A - mixed-source landmark progression")
    for r in data["task_a"]:
        ci = r["roc_auc_ci"]
        crosses = " CROSSES 0.50" if ci and ci[0] <= 0.5 <= ci[1] else ""
        print(
            f"  {r['label']:<20} n={r['n']:<5} AUC={r['roc_auc']:.3f} "
            f"[{ci[0]:.3f}, {ci[1]:.3f}]  ECE={r['ece']:.3f}{crosses}"
        )
    print("\nTask B - C-index collapse (mixed -> within-source)")
    for c in data["task_b_collapse"]:
        ci = c["within_ci"]
        print(
            f"  {c['label']:<20} {c['mixed']:.3f} -> {c['within']:.3f} "
            f"({c['delta']:+.3f})  within-CI [{ci[0]:.3f}, {ci[1]:.3f}]"
        )
    n_cross = sum(
        1
        for r in data["task_a"]
        if r["roc_auc_ci"] and r["roc_auc_ci"][0] <= 0.5 <= r["roc_auc_ci"][1]
    )
    print(f"\n  {n_cross}/{len(data['task_a'])} Task A intervals contain 0.50")
    for u in data["meta"]["unbacked_claims"]:
        print(f"\n  ! UNBACKED: {u['claim']}\n      {u['status']}")


def main():
    if not os.path.isdir(REPORTS):
        print(f"missing {REPORTS}", file=sys.stderr)
        return 1
    data = build()
    report(data)

    if "--check" in sys.argv:
        ok = bool(data["task_a"]) and bool(data["task_b_collapse"])
        missing = [r["label"] for r in data["task_a"] if not r["roc_auc_ci"]]
        if missing:
            print(f"\n  ! Task A rows without CI: {missing}")
            ok = False
        print("\n  check: PASS" if ok else "\n  check: FAIL")
        return 0 if ok else 1

    out_path = DEFAULT_OUT
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            out_path = a
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print(f"\nSaved -> {out_path}  ({os.path.getsize(out_path) / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
