#!/usr/bin/env python3
"""Set wearing_style.sleeve_state=not_applicable for sleeveless regions only."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from annotate_api import read_jsonl, write_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "annotations_api.jsonl"
REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix sleeve_state to not_applicable when sleeve_length is sleeveless."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def is_sleeveless(region: dict[str, Any]) -> bool:
    local_structure = region.get("local_structure")
    if not isinstance(local_structure, dict):
        return False
    return local_structure.get("sleeve_length") == "sleeveless"


def fix_sleeve_state_in_region(region: dict[str, Any]) -> bool:
    if not is_sleeveless(region):
        return False
    wearing_style = region.setdefault("wearing_style", {})
    if wearing_style.get("sleeve_state") == "not_applicable":
        return False
    wearing_style["sleeve_state"] = "not_applicable"
    return True


def fix_row(row: dict[str, Any]) -> int:
    changed = 0
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        return 0

    person_image = annotation.get("person_image")
    if isinstance(person_image, dict):
        person_annotation = person_image.get("person_annotation")
        if isinstance(person_annotation, dict):
            for region_key in REGION_KEYS:
                region = person_annotation.get(region_key)
                if isinstance(region, dict) and region.get("is_present") is not False:
                    if fix_sleeve_state_in_region(region):
                        changed += 1

    garment_image = annotation.get("garment_image")
    if isinstance(garment_image, dict):
        garment_annotation = garment_image.get("garment_annotation")
        if isinstance(garment_annotation, dict):
            for region_key in REGION_KEYS:
                region = garment_annotation.get(region_key)
                if isinstance(region, dict):
                    if fix_sleeve_state_in_region(region):
                        changed += 1
    return changed


def main() -> int:
    args = parse_args()
    rows = read_jsonl(args.input_jsonl)
    updated_rows: list[dict[str, Any]] = []
    total_changes = 0
    touched_samples: list[str] = []

    for row in rows:
        merged = copy.deepcopy(row)
        n = fix_row(merged)
        if n:
            total_changes += n
            touched_samples.append(str(row.get("source_sample_id", "unknown")))
        updated_rows.append(merged)

    print(f"input={args.input_jsonl}", flush=True)
    print(f"rows={len(rows)} regions_changed={total_changes} samples={len(touched_samples)}", flush=True)
    if touched_samples:
        print("touched:", ", ".join(touched_samples), flush=True)

    if args.dry_run:
        print("dry_run=true no file written", flush=True)
        return 0

    write_jsonl(args.input_jsonl, updated_rows)
    print(f"written={args.input_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
