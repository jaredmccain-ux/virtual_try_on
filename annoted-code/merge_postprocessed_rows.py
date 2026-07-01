#!/usr/bin/env python3
"""Post-process a subset of rows and merge back into a full postprocessed JSONL."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from annotate_api import read_jsonl, write_jsonl  # noqa: E402
from postprocess_annotations import postprocess_row  # noqa: E402

DEFAULT_SOURCE = SCRIPT_DIR / "annotations_api.jsonl"
DEFAULT_MERGE_INTO = SCRIPT_DIR / "annotations_api_postprocessed.jsonl"

KNOWN_MISSING_IDS = (
    "upper_body_048453",
    "lower_body_050192",
    "lower_body_050193",
    "lower_body_050209",
    "lower_body_050225",
    "lower_body_050226",
    "lower_body_050233",
    "lower_body_050234",
    "lower_body_050253",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-process selected rows and merge into postprocessed JSONL."
    )
    parser.add_argument("--source-jsonl", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--merge-into", type=Path, default=DEFAULT_MERGE_INTO)
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated source_sample_ids (default: known 9 missing-garment ids).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_ids(ids_arg: str) -> list[str]:
    if not ids_arg.strip():
        return list(KNOWN_MISSING_IDS)
    return [part.strip() for part in ids_arg.split(",") if part.strip()]


def main() -> int:
    args = parse_args()
    ids = parse_ids(args.ids)
    id_set = set(ids)

    if not args.source_jsonl.is_file():
        print(f"ERROR: source not found: {args.source_jsonl}", flush=True)
        return 1
    if not args.merge_into.is_file():
        print(f"ERROR: merge target not found: {args.merge_into}", flush=True)
        return 1

    source_rows = read_jsonl(args.source_jsonl)
    source_by_id = {str(r["source_sample_id"]): r for r in source_rows}

    missing = [sid for sid in ids if sid not in source_by_id]
    if missing:
        print(f"ERROR: ids not in source: {missing}", flush=True)
        return 1

    processed = {sid: postprocess_row(copy.deepcopy(source_by_id[sid])) for sid in ids}

    target_rows = read_jsonl(args.merge_into)
    updated = 0
    for index, row in enumerate(target_rows):
        sid = str(row.get("source_sample_id"))
        if sid in processed:
            target_rows[index] = processed[sid]
            updated += 1

    not_in_target = [sid for sid in ids if sid not in {str(r.get("source_sample_id")) for r in target_rows}]
    if not_in_target:
        print(f"WARNING: ids not in merge target (skipped): {not_in_target}", flush=True)

    print(f"postprocessed {len(processed)} rows, merged {updated} into {args.merge_into}", flush=True)

    if args.dry_run:
        print("dry_run=true no file written", flush=True)
        return 0

    write_jsonl(args.merge_into, target_rows)
    print(f"written: {args.merge_into}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
