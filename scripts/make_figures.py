"""Generate the v1 paper figures.

PAPER_OUTLINE.md does not literally enumerate a "Figures" section; the four
figures below are inferred from its Results (§9) and Discussion (§11) text:

  Figure 1: Headline Task A discrimination + Brier per baseline (§9.1).
  Figure 2: Reliability diagrams per baseline on Task A (§9.3).
  Figure 3: Cohort composition by source x EGFR variant class (§4).
  Figure 4: Discrimination vs calibration trade-off scatter (§11 thesis).

All four read from `eval_reports/*.json` and the parquet at
`data/processed/oncotraj_v0/`. Each is a standalone function that takes a
dict of report objects + (where relevant) the parquet path, and writes a
matched pair of PDF + PNG at 300 dpi to `paper/figures/`. `paper/` is
gitignored on this repo, so the artefacts stay local.

Style: sans-serif (Helvetica/Arial fallback), Wong/Bang-Wong-inspired
colorblind-friendly palette, no top/right spines, no chartjunk.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_REPORTS_DIR = REPO_ROOT / "eval_reports"
DEFAULT_PARQUET_DIR = REPO_ROOT / "data" / "processed" / "oncotraj_v0"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "figures"

# Bang Wong (2011 Nature Methods) colorblind-friendly palette.
PALETTE = {
    "majority": "#999999",
    "logistic": "#E69F00",
    "random_forest": "#56B4E9",
    "xgboost": "#009E73",
    "lstm": "#F0E442",
    "transformer": "#0072B2",
    "accent": "#D55E00",
    "neutral": "#CC79A7",
}

MODEL_DISPLAY = {
    "majority": "Majority",
    "logistic": "Logistic",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lstm": "LSTM",
    "transformer": "Transformer",
}


def set_publication_style() -> None:
    """Apply a restrained, publication-friendly matplotlib style."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "axes.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.dpi": 100,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.transparent": False,
            "pdf.fonttype": 42,  # editable vector text in Illustrator.
            "ps.fonttype": 42,
        }
    )


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------


def load_task_a_reports(reports_dir: Path) -> dict[str, dict]:
    """Return {model_name: report_dict} for every Task-A eval JSON found."""
    out: dict[str, dict] = {}
    for path in sorted(reports_dir.glob("*_taskA.json")):
        data = json.loads(path.read_text())
        if data.get("task_a"):
            name = data["submission_id"].removesuffix("_taskA")
            out[name] = data
    return out


def _save(fig: plt.Figure, name: str, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{name}.pdf"
    png = out_dir / f"{name}.png"
    fig.savefig(pdf)
    fig.savefig(png)
    plt.close(fig)
    return pdf, png


# ---------------------------------------------------------------------------
# Figure 1 — Headline Task A discrimination + Brier per baseline
# ---------------------------------------------------------------------------


def figure_1_headline(
    reports: dict[str, dict], out_dir: Path = DEFAULT_OUT_DIR
) -> tuple[Path, Path]:
    """Two-panel bar chart: ROC-AUC and Brier on Task A test split."""
    names = sorted(reports.keys(), key=lambda n: reports[n]["task_a"].get("roc_auc") or 0)
    aucs = [reports[n]["task_a"].get("roc_auc") for n in names]
    briers = [reports[n]["task_a"].get("brier") for n in names]
    colors = [PALETTE.get(n, "#444444") for n in names]
    labels = [MODEL_DISPLAY.get(n, n) for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=False)
    y = np.arange(len(names))

    axes[0].barh(y, aucs, color=colors, edgecolor="white", linewidth=1.2)
    axes[0].axvline(0.5, color="#555555", lw=0.7, ls="--", zorder=0)
    axes[0].set_xlim(0.4, 1.0)
    axes[0].set_xlabel("ROC-AUC")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].set_title("a · Discrimination (higher is better)", loc="left")
    for i, v in enumerate(aucs):
        if v is not None:
            axes[0].text(v + 0.005, i, f"{v:.3f}", va="center", ha="left", fontsize=8)

    axes[1].barh(y, briers, color=colors, edgecolor="white", linewidth=1.2)
    axes[1].set_xlim(0, max([b for b in briers if b is not None]) * 1.25)
    axes[1].set_xlabel("Brier score")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].set_title("b · Calibration error (lower is better)", loc="left")
    for i, v in enumerate(briers):
        if v is not None:
            axes[1].text(
                v + axes[1].get_xlim()[1] * 0.01, i, f"{v:.3f}", va="center", ha="left", fontsize=8
            )

    n_test = next(
        (reports[n]["task_a"].get("n") for n in names if reports[n]["task_a"].get("n")),
        None,
    )
    n_label = f" · n={int(n_test)}" if n_test else ""
    fig.suptitle(f"Task A (12-month landmark) · test split{n_label}", y=1.02, fontsize=10)
    return _save(fig, "figure_1_headline", out_dir)


# ---------------------------------------------------------------------------
# Figure 2 — Reliability diagrams per baseline
# ---------------------------------------------------------------------------


def figure_2_reliability(
    reports: dict[str, dict], out_dir: Path = DEFAULT_OUT_DIR
) -> tuple[Path, Path]:
    """Grid of reliability diagrams (one panel per baseline)."""
    names = sorted(
        reports.keys(), key=lambda n: reports[n]["task_a"].get("roc_auc") or 0, reverse=True
    )
    n = len(names)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.0 * ncols, 2.2 * nrows + 0.4), sharex=True, sharey=True
    )
    axes = np.atleast_2d(axes)

    for i, name in enumerate(names):
        rel = reports[name]["task_a"].get("reliability") or {}
        ax = axes[i // ncols, i % ncols]
        ax.plot([0, 1], [0, 1], color="#888888", lw=0.7, ls="--", zorder=1)
        centers = rel.get("bin_centers", [])
        freqs = rel.get("frequencies", [])
        counts = rel.get("counts", [])
        if centers:
            sizes = np.array(counts) * 12 + 8
            ax.scatter(
                centers,
                freqs,
                s=sizes,
                color=PALETTE.get(name, "#444444"),
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            ax.plot(centers, freqs, color=PALETTE.get(name, "#444444"), lw=1.0, zorder=2)
        ece = reports[name]["task_a"].get("ece")
        ax.set_title(
            f"{MODEL_DISPLAY.get(name, name)}\nECE = {ece:.3f}"
            if ece is not None
            else MODEL_DISPLAY.get(name, name),
            loc="left",
            fontsize=9,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        if i % ncols == 0:
            ax.set_ylabel("Observed frequency")
        if i // ncols == nrows - 1:
            ax.set_xlabel("Predicted probability")

    # Hide unused axes if any.
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].axis("off")

    fig.suptitle("Task A (12-month landmark) reliability · test split", y=1.02, fontsize=10)
    return _save(fig, "figure_2_reliability", out_dir)


# ---------------------------------------------------------------------------
# Figure 3 — Cohort composition
# ---------------------------------------------------------------------------


def figure_3_cohort(
    parquet_dir: Path = DEFAULT_PARQUET_DIR, out_dir: Path = DEFAULT_OUT_DIR
) -> tuple[Path, Path]:
    """Two-panel: stacked bar of source by EGFR variant class + cohort totals."""
    patients = pd.read_parquet(parquet_dir / "patients.parquet")
    cohort = patients.loc[patients["included_in_v1_cohort"]].copy()
    if cohort.empty:
        cohort = patients.copy()

    crosstab = pd.crosstab(cohort["source_dataset"], cohort["egfr_variant_class"])
    classes = list(crosstab.columns)
    sources = list(crosstab.index)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), gridspec_kw={"width_ratios": [2.0, 1.0]})

    # Panel a: stacked horizontal bars.
    cls_colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(classes)))
    left = np.zeros(len(sources))
    for cls, color in zip(classes, cls_colors, strict=False):
        widths = crosstab[cls].to_numpy()
        axes[0].barh(
            sources, widths, left=left, color=color, edgecolor="white", linewidth=0.8, label=cls
        )
        left += widths
    axes[0].set_xlabel("Patients (n)")
    axes[0].set_title("a · Cohort by source by EGFR variant class", loc="left")
    axes[0].legend(
        loc="lower right",
        title="EGFR class",
        title_fontsize=8,
        ncol=1,
        bbox_to_anchor=(1.0, -0.05),
        borderaxespad=0,
    )
    for i, src in enumerate(sources):
        total = crosstab.loc[src].sum()
        axes[0].text(total + 1, i, f"{total}", va="center", ha="left", fontsize=8)

    # Panel b: pie chart of source totals.
    totals = crosstab.sum(axis=1)
    src_colors = [
        PALETTE.get("logistic"),
        PALETTE.get("random_forest"),
        PALETTE.get("transformer"),
        PALETTE.get("accent"),
        PALETTE.get("neutral"),
    ][: len(totals)]
    axes[1].pie(
        totals.to_numpy(),
        labels=totals.index,
        colors=src_colors,
        autopct=lambda p: f"{round(p * totals.sum() / 100)}",
        textprops={"fontsize": 8},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        startangle=90,
        counterclock=False,
    )
    axes[1].set_title(f"b · Total: n={int(totals.sum())}", loc="left")

    return _save(fig, "figure_3_cohort", out_dir)


# ---------------------------------------------------------------------------
# Figure 4: Discrimination vs calibration trade-off
# ---------------------------------------------------------------------------


def figure_4_tradeoff(
    reports: dict[str, dict], out_dir: Path = DEFAULT_OUT_DIR
) -> tuple[Path, Path]:
    """ROC-AUC vs ECE scatter, one point per baseline.

    The §11 thesis is that the best-discrimination model is rarely the
    best-calibrated. This scatter surfaces that trade-off in one panel.
    """
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for name, data in reports.items():
        ta = data.get("task_a", {})
        auc = ta.get("roc_auc")
        ece = ta.get("ece")
        if auc is None or ece is None:
            continue
        ax.scatter(
            ece,
            auc,
            s=180,
            color=PALETTE.get(name, "#444444"),
            edgecolor="white",
            linewidth=1.2,
            zorder=3,
        )
        ax.annotate(
            MODEL_DISPLAY.get(name, name),
            (ece, auc),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=9,
        )
    # Reference lines.
    ax.axhline(0.5, color="#888888", lw=0.6, ls="--", zorder=1)
    ax.axvline(0.0, color="#888888", lw=0.6, ls="--", zorder=1)
    ax.set_xlabel("Expected Calibration Error  (lower is better)")
    ax.set_ylabel("ROC-AUC  (higher is better)")
    ax.set_title("Task A · discrimination vs calibration", loc="left")

    # "Pareto" hint: model in the top-left corner is best on both.
    ax.annotate(
        "ideal corner",
        xy=(0.02, 0.96),
        xycoords="axes fraction",
        xytext=(0.18, 0.86),
        textcoords="axes fraction",
        fontsize=8,
        color="#555555",
        style="italic",
        arrowprops={"arrowstyle": "->", "color": "#999999", "lw": 0.6, "shrinkA": 0, "shrinkB": 0},
    )
    # Generous padding so labels don't clip.
    xs = [r["task_a"]["ece"] for r in reports.values() if r["task_a"].get("ece") is not None]
    ys = [
        r["task_a"]["roc_auc"] for r in reports.values() if r["task_a"].get("roc_auc") is not None
    ]
    if xs and ys:
        ax.set_xlim(-0.01, max(xs) * 1.3)
        ax.set_ylim(min(0.45, min(ys) - 0.05), 1.0)

    return _save(fig, "figure_4_tradeoff", out_dir)


# ---------------------------------------------------------------------------
# Figure 5 — v1.1 cross-source Task B generalization (headline figure)
# ---------------------------------------------------------------------------


def _load_v11_cross_source_reports(reports_dir: Path) -> list[dict]:
    """Return the v11_*_<train>_<eval>.json reports as a flat list."""
    rows: list[dict] = []
    for path in sorted(reports_dir.glob("v11_*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        meta = data.get("v11_meta") or {}
        tb = data.get("task_b") or {}
        rows.append(
            {
                "model": meta.get("model"),
                "train_src": meta.get("train_source"),
                "eval_src": meta.get("eval_source"),
                "eval_label": meta.get("eval_label"),
                "mae": tb.get("mae"),
                "mae_ci": tb.get("mae_ci"),
                "c_index": tb.get("c_index"),
                "c_index_ci": tb.get("c_index_ci"),
                "n_eval": tb.get("n"),
                "n_obs": tb.get("n_observed"),
            }
        )
    return rows


def figure_5_cross_source(
    reports_dir: Path = DEFAULT_REPORTS_DIR, out_dir: Path = DEFAULT_OUT_DIR
) -> tuple[Path, Path]:
    """Headline v1.1 figure: in-distribution vs cross-source Task B MAE per model.

    Grouped bars; one bar per (model, eval target). Error bars show the
    95% bootstrap CI from the eval report. FLAURA is rendered with a
    hatched pattern and a footnote so the constant-pseudo-target artifact
    is impossible to miss.
    """
    rows = _load_v11_cross_source_reports(reports_dir)
    if not rows:
        raise RuntimeError(
            f"No v11_*.json reports found in {reports_dir}; "
            "run scripts/train_v11_cross_source.py first."
        )

    models_in_order = ["majority", "logistic", "random_forest", "xgboost", "cox"]
    eval_targets = [
        ("MSK_CHORD", "MSK-CHORD\n(in-distribution test)", "#0E7C66", False),
        ("GENIE_BPC", "GENIE BPC\n(cross-source)", "#0072B2", False),
        ("FLAURA_SUPP", "FLAURA supplement\n(cross-source, const. target†)", "#D55E00", True),
    ]

    by_key = {(r["model"], r["eval_src"]): r for r in rows if r["mae"] is not None}

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    n_models = len(models_in_order)
    n_targets = len(eval_targets)
    bar_w = 0.78 / n_targets
    xs = np.arange(n_models)

    for i, (src, label, color, hatched) in enumerate(eval_targets):
        offset = (i - (n_targets - 1) / 2) * bar_w
        heights = []
        err_lo = []
        err_hi = []
        for m in models_in_order:
            r = by_key.get((m, src))
            if r is None or r["mae"] is None:
                heights.append(0)
                err_lo.append(0)
                err_hi.append(0)
                continue
            heights.append(r["mae"])
            ci = r["mae_ci"]
            if ci:
                err_lo.append(r["mae"] - ci[0])
                err_hi.append(ci[1] - r["mae"])
            else:
                err_lo.append(0)
                err_hi.append(0)
        ax.bar(
            xs + offset,
            heights,
            bar_w,
            color=color,
            edgecolor="white",
            linewidth=0.6,
            label=label,
            hatch="///" if hatched else None,
        )
        ax.errorbar(
            xs + offset,
            heights,
            yerr=[err_lo, err_hi],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.7,
            capsize=2.5,
            zorder=4,
        )
        for x, h in zip(xs + offset, heights, strict=False):
            if h > 0:
                ax.text(x, h + 12, f"{round(h)}", ha="center", fontsize=7, color="#333333")

    ax.set_xticks(xs)
    ax.set_xticklabels([MODEL_DISPLAY.get(m, m) for m in models_in_order], fontsize=9)
    ax.set_ylabel("Task B test MAE (days; lower is better)")
    ax.set_title(
        "Task B (time-to-progression) · in-distribution vs cross-source · "
        "all models trained on MSK-CHORD only",
        loc="left",
    )
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        frameon=False,
        fontsize=8,
    )
    # Footnote on FLAURA constant target.
    fig.text(
        0.01,
        -0.01,
        "† FLAURA supplement has a single pseudo-PFS value (910 d) for every patient "
        "in the harmonization layer, so MAE here is dominated by that fixed offset "
        "and C-index is undefined.",
        ha="left",
        fontsize=7,
        color="#555555",
        style="italic",
    )

    fig.tight_layout()
    return _save(fig, "figure_5_cross_source", out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def make_all_figures(
    reports_dir: Path = DEFAULT_REPORTS_DIR,
    parquet_dir: Path = DEFAULT_PARQUET_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    only: Iterable[int] | None = None,
) -> list[tuple[str, Path, Path]]:
    set_publication_style()
    reports = load_task_a_reports(reports_dir)
    only_set = set(only) if only else {1, 2, 3, 4, 5}
    results: list[tuple[str, Path, Path]] = []
    if 1 in only_set:
        pdf, png = figure_1_headline(reports, out_dir)
        results.append(("figure_1_headline", pdf, png))
    if 2 in only_set:
        pdf, png = figure_2_reliability(reports, out_dir)
        results.append(("figure_2_reliability", pdf, png))
    if 3 in only_set:
        pdf, png = figure_3_cohort(parquet_dir, out_dir)
        results.append(("figure_3_cohort", pdf, png))
    if 4 in only_set:
        pdf, png = figure_4_tradeoff(reports, out_dir)
        results.append(("figure_4_tradeoff", pdf, png))
    if 5 in only_set:
        try:
            pdf, png = figure_5_cross_source(reports_dir, out_dir)
            results.append(("figure_5_cross_source", pdf, png))
        except RuntimeError as exc:
            print(f"figure_5_cross_source: skipped — {exc}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--parquet-dir", type=Path, default=DEFAULT_PARQUET_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--only",
        type=int,
        nargs="*",
        choices=[1, 2, 3, 4, 5],
        help="Restrict to a subset of figures (default: all 5).",
    )
    args = parser.parse_args()

    written = make_all_figures(
        reports_dir=args.reports_dir,
        parquet_dir=args.parquet_dir,
        out_dir=args.out_dir,
        only=args.only,
    )
    for name, pdf, png in written:
        print(f"{name}: {pdf} {png}")


if __name__ == "__main__":
    main()
