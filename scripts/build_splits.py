"""Build the train/val/test split manifest from an existing OncoTraj parquet.

Usage:
    python scripts/build_splits.py \
        --parquet data/processed/oncotraj_v0 \
        --output  data/processed/oncotraj_v0/_splits.json \
        --seed    42

The output JSON conforms to `oncotraj-splits/1.0.0`; see
`src/oncotraj/data/splits.py` for the schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from oncotraj.data.splits import (  # noqa: E402
    DEFAULT_MIN_STRATUM_SIZE,
    DEFAULT_PROPORTIONS,
    DEFAULT_SEED,
    SplitProportions,
    make_splits_from_parquet,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        required=True,
        type=Path,
        help="Directory containing patients.parquet and outcomes.parquet.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Output JSON manifest path.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--proportions",
        nargs=3,
        type=float,
        metavar=("TRAIN", "VAL", "TEST"),
        default=list(DEFAULT_PROPORTIONS),
    )
    parser.add_argument(
        "--site-out",
        nargs="*",
        default=[],
        help="Source datasets to hold out entirely (e.g. TRACERX).",
    )
    parser.add_argument("--min-stratum-size", type=int, default=DEFAULT_MIN_STRATUM_SIZE)
    parser.add_argument(
        "--strata-fields", nargs="*", default=["source_dataset", "resistance_mechanism"]
    )
    args = parser.parse_args()

    manifest = make_splits_from_parquet(
        parquet_dir=args.parquet,
        output_path=args.output,
        seed=args.seed,
        proportions=SplitProportions(*args.proportions),
        site_out=args.site_out,
        min_stratum_size=args.min_stratum_size,
        strata_fields=args.strata_fields,
    )

    print(f"Wrote split manifest to {args.output}")
    print(
        f"  train={manifest.counts['train']}  val={manifest.counts['val']}  test={manifest.counts['test']}"
    )
    print(f"  excluded (held-out or site-out): {manifest.excluded_count}")


if __name__ == "__main__":
    main()
