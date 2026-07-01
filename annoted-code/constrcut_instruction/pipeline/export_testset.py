#!/usr/bin/env python3
"""Export FireRed flat testset rows from with_edit JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from edit_common import read_jsonl, sample_key, write_jsonl


PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
DATA_DIR = ROOT_DIR / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FireRed testset JSONL.")
    parser.add_argument(
        "--dataset",
        choices=["paired", "unpair", "both"],
        default="both",
    )
    parser.add_argument(
        "--paired-input",
        type=Path,
        default=DATA_DIR / "annotations_paired_with_edit.jsonl",
    )
    parser.add_argument(
        "--unpair-input",
        type=Path,
        default=DATA_DIR / "annotations_unpair_with_edit.jsonl",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=DATA_DIR / "testset_paired.jsonl",
    )
    parser.add_argument(
        "--unpair-output",
        type=Path,
        default=DATA_DIR / "testset_unpair.jsonl",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def flatten_row(row: dict[str, Any]) -> dict[str, Any] | None:
    edit_task = row.get("edit_task")
    instruction = row.get("instruction")
    if not isinstance(edit_task, dict) or not isinstance(instruction, dict):
        return None

    return {
        "sample_id": row.get("sample_id") or sample_key(row),
        "source_image_path": row.get("person_image"),
        "garment_image_path": row.get("garment_image"),
        "instruction_en": instruction.get("instruction_en"),
        "edit_type_id": edit_task.get("edit_type_id"),
        "pair_mode": row.get("pair_mode") or "paired",
    }


def export_dataset(
    *,
    pair_mode: str,
    input_path: Path,
    output_path: Path,
    overwrite: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite")

    rows = read_jsonl(input_path)
    flat_rows: list[dict[str, Any]] = []
    skipped = 0
    for row in rows:
        flat = flatten_row(row)
        if flat is None:
            skipped += 1
            continue
        flat_rows.append(flat)

    write_jsonl(output_path, flat_rows)
    print(f"\n=== {pair_mode} testset ===")
    print(f"input={input_path}")
    print(f"rows={len(flat_rows)} skipped={skipped}")
    print(f"output={output_path}")


def main() -> int:
    args = parse_args()
    if args.dataset in ("paired", "both"):
        export_dataset(
            pair_mode="paired",
            input_path=args.paired_input,
            output_path=args.paired_output,
            overwrite=args.overwrite,
        )
    if args.dataset in ("unpair", "both"):
        export_dataset(
            pair_mode="unpair",
            input_path=args.unpair_input,
            output_path=args.unpair_output,
            overwrite=args.overwrite,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
