"""`oncotraj-eval` CLI entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .leaderboard import update_leaderboard
from .report import SCHEMA_DOC, evaluate, load_predictions


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="oncotraj-eval",
        description="Evaluate a predictions CSV against the locked OncoTraj v1 metrics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=SCHEMA_DOC,
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help="Path to predictions.csv. Required unless --refresh is used.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=("train", "val", "test"),
        help="Split to evaluate against (default: test).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Where to write the JSON report. Defaults to "
        "<reports-dir>/<submission-id>.json when --reports-dir is set.",
    )
    parser.add_argument(
        "--submission-id",
        help="Identifier for the leaderboard row. Defaults to the CSV stem.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("eval_reports"),
        help="Directory where JSON reports live; used by --leaderboard "
        "and as the default output dir when --output is omitted.",
    )
    parser.add_argument(
        "--leaderboard",
        type=Path,
        help="If set, rebuild the markdown leaderboard at this path after writing the JSON report.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Rebuild the leaderboard from existing JSON reports without "
        "running a fresh evaluation. Requires --leaderboard.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.refresh:
        if args.leaderboard is None:
            parser.error("--refresh requires --leaderboard.")
        path = update_leaderboard(args.reports_dir, args.leaderboard)
        print(f"Refreshed leaderboard at {path}")
        return 0

    if args.predictions is None:
        parser.error("--predictions is required unless --refresh is given.")

    submission_id = args.submission_id or args.predictions.stem
    output = args.output or (args.reports_dir / f"{submission_id}.json")

    predictions = load_predictions(args.predictions)
    report = evaluate(predictions, split=args.split, submission_id=submission_id)
    report.write(output)
    print(f"Wrote eval report to {output}")
    print(f"Tasks evaluated: {', '.join(report.tasks_evaluated)}")

    if args.leaderboard is not None:
        path = update_leaderboard(args.reports_dir, args.leaderboard)
        print(f"Updated leaderboard at {path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
