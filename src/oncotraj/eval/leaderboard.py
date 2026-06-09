"""Auto-update a markdown leaderboard from JSON eval reports.

The leaderboard file has hand-written prose plus an auto-generated table
region delimited by markers. `update_leaderboard()` reads every `*.json`
report in a directory, rewrites the region between the markers, and
preserves everything else. The default sort is Task A ROC-AUC on test
(highest first), with secondary sort by Task C macro F1.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

START_MARKER = "<!-- LEADERBOARD START -->"
END_MARKER = "<!-- LEADERBOARD END -->"

DEFAULT_TEMPLATE = """\
# OncoTraj v1 Leaderboard

Auto-generated from `src/oncotraj/eval`. Submissions are evaluated against
the locked test split via `oncotraj-eval --predictions <csv> --split test`
and dropped as JSON into `eval_reports/`. Run `oncotraj-eval --refresh`
(or call `oncotraj.eval.update_leaderboard(...)` programmatically) to
regenerate the table below.

Default sort: **Task A ROC-AUC on test**, descending. Secondary sort:
**Task C macro F1**.

{START}
{END}

## How to submit

1. Train a model on the OncoTraj v1 cohort (see `data/processed/oncotraj_v0/`).
2. Write predictions to a CSV conforming to the schema in
   `src/oncotraj/eval/report.py` (`SCHEMA_DOC`).
3. Run `oncotraj-eval --predictions yours.csv --split test \\
       --submission-id <your-name> --output eval_reports/<your-name>.json`.
4. Rerun with `--leaderboard leaderboard.md` to update this file.
"""


@dataclass
class LeaderboardRow:
    submission_id: str
    split: str
    task_a_auc: float | None
    task_a_brier: float | None
    task_a_ece: float | None
    task_b_mae: float | None
    task_b_c_index: float | None
    task_c_macro_f1: float | None
    task_c_accuracy: float | None
    n_predictions: int
    generated_at_utc: str

    @classmethod
    def from_report(cls, data: dict) -> LeaderboardRow:
        task_a = data.get("task_a") or {}
        task_b = data.get("task_b") or {}
        task_c = data.get("task_c") or {}
        return cls(
            submission_id=str(data.get("submission_id", "anonymous")),
            split=str(data.get("split", "")),
            task_a_auc=task_a.get("roc_auc"),
            task_a_brier=task_a.get("brier"),
            task_a_ece=task_a.get("ece"),
            task_b_mae=task_b.get("mae"),
            task_b_c_index=task_b.get("c_index"),
            task_c_macro_f1=task_c.get("macro_f1"),
            task_c_accuracy=task_c.get("accuracy"),
            n_predictions=int(data.get("n_predictions", 0)),
            generated_at_utc=str(data.get("generated_at_utc", "")),
        )


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None or (isinstance(v, float) and (v != v)):  # NaN check
        return "—"
    return f"{v:.{digits}f}"


def _render_table(rows: list[LeaderboardRow], split: str) -> str:
    rows = [r for r in rows if r.split == split]
    rows.sort(
        key=lambda r: (
            -(r.task_a_auc if r.task_a_auc is not None else -1.0),
            -(r.task_c_macro_f1 if r.task_c_macro_f1 is not None else -1.0),
        )
    )
    if not rows:
        return f"_No submissions on the `{split}` split yet._"

    header = (
        "| Submission | n | Task A AUC | Task A Brier | Task A ECE | "
        "Task B MAE (d) | Task B C-index | Task C macro F1 | Task C accuracy | Generated |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    body = [
        f"| `{r.submission_id}` | {r.n_predictions} | "
        f"{_fmt(r.task_a_auc)} | {_fmt(r.task_a_brier)} | {_fmt(r.task_a_ece)} | "
        f"{_fmt(r.task_b_mae, 1)} | {_fmt(r.task_b_c_index)} | "
        f"{_fmt(r.task_c_macro_f1)} | {_fmt(r.task_c_accuracy)} | "
        f"{r.generated_at_utc[:19]} |"
        for r in rows
    ]
    return f"### Split: `{split}`\n\n" + "\n".join([header, sep, *body])


def _load_rows(reports_dir: Path) -> list[LeaderboardRow]:
    rows: list[LeaderboardRow] = []
    for path in sorted(reports_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if not str(data.get("schema_version", "")).startswith("oncotraj-eval/"):
            continue
        # v1.1 cross-source reports have their own dedicated section in
        # leaderboard.md and would clutter the headline table — skip them
        # here.
        if str(data.get("submission_id", "")).startswith("v11_"):
            continue
        rows.append(LeaderboardRow.from_report(data))
    return rows


def update_leaderboard(
    reports_dir: str | Path,
    leaderboard_path: str | Path,
    splits: list[str] | None = None,
) -> Path:
    """Rebuild the auto-generated region of the leaderboard markdown.

    Existing content outside the markers is preserved; the region between
    the markers is overwritten. A missing leaderboard file is created from
    `DEFAULT_TEMPLATE`.
    """
    reports_dir = Path(reports_dir)
    leaderboard_path = Path(leaderboard_path)
    splits = splits or ["test", "val"]

    rows = _load_rows(reports_dir)
    tables = [_render_table(rows, s) for s in splits]
    auto_region = "\n\n".join(tables)
    block = f"{START_MARKER}\n<!-- Auto-generated; do not edit between markers. -->\n\n{auto_region}\n\n{END_MARKER}"

    if leaderboard_path.exists():
        text = leaderboard_path.read_text()
        if START_MARKER in text and END_MARKER in text:
            new_text = re.sub(
                rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
                block,
                text,
                flags=re.DOTALL,
            )
        else:
            new_text = text.rstrip() + f"\n\n{block}\n"
    else:
        new_text = DEFAULT_TEMPLATE.format(START=START_MARKER, END=END_MARKER)
        new_text = re.sub(
            rf"{re.escape(START_MARKER)}.*?{re.escape(END_MARKER)}",
            block,
            new_text,
            flags=re.DOTALL,
        )

    leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    leaderboard_path.write_text(new_text)
    return leaderboard_path
