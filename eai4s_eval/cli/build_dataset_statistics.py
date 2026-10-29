from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from eai4s_eval.dataset_statistics import (
    DEFAULT_JSON_PATH,
    DEFAULT_WORKBOOK_PATH,
    DatasetStatisticsValidationError,
    generate_statistics_json,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the unified dataset registry and generate dashboard JSON."
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=DEFAULT_WORKBOOK_PATH,
        help="Path to dataset_statistics_registry.xlsx.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help="Path to the generated dataset_statistics.json.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate the workbook/formulas and fail if JSON is missing or out of sync.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = generate_statistics_json(
            workbook_path=args.workbook,
            output_path=args.output,
            check=args.check,
        )
    except (DatasetStatisticsValidationError, RuntimeError) as exc:
        print(f"dataset statistics validation failed: {exc}", file=sys.stderr)
        return 1
    if args.check:
        print(f"dataset statistics are valid and synchronized: {args.output}")
    elif result.changed:
        print(f"generated dataset statistics: {args.output}")
    else:
        print(f"dataset statistics unchanged: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
