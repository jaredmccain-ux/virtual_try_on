#!/usr/bin/env python3
"""Label paired samples with scene_id from person + garment annotations."""

from __future__ import annotations

import argparse
import copy
from collections import Counter
from pathlib import Path
from typing import Any

from edit_common import get_person_annotation, read_jsonl, write_jsonl
from scene_classify import (
    active_region_from_garment,
    classify_paired_scene,
    garment_target,
    person_wearing_type,
)

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
ANNOTED_CODE_DIR = ROOT_DIR.parent
DEFAULT_INPUT = ANNOTED_CODE_DIR / "annotations_rivo_reviewed_supplement_checkpoint_postprocessed.jsonl"
DEFAULT_OUTPUT = DATA_DIR / "annotations_paired_scenes.jsonl"


def get_garment_annotation_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row["annotation"]["garment_image"]["garment_annotation"]


def label_row(row: dict[str, Any]) -> dict[str, Any]:
    labeled = copy.deepcopy(row)
    person_annotation = get_person_annotation(row)
    garment_annotation = get_garment_annotation_from_row(row)
    scene_id = classify_paired_scene(person_annotation, garment_annotation)
    if scene_id is None:
        raise ValueError(f"{row.get('source_sample_id')}: unable to classify paired scene")

    wear_kind, upper_state = person_wearing_type(person_annotation)
    where, garment_class, _ = garment_target(person_annotation, garment_annotation)

    labeled["pair_mode"] = "paired"
    labeled["scene_id"] = scene_id
    labeled["person_upper_state"] = upper_state or wear_kind
    labeled["garment_class"] = "G0" if scene_id == "C" else garment_class
    labeled["active_region"] = active_region_from_garment(garment_annotation)
    labeled["source_attributes"] = person_annotation
    if not labeled.get("sample_id"):
        labeled["sample_id"] = str(labeled.get("source_sample_id", "unknown"))
    return labeled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label paired rows with scene_id.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"input not found: {args.input_jsonl}")
    if args.output_jsonl.exists() and not args.overwrite and not args.dry_run:
        raise FileExistsError(f"{args.output_jsonl} exists; pass --overwrite")

    rows = read_jsonl(args.input_jsonl)
    labeled_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    scene_counts: Counter[str] = Counter()

    for row in rows:
        try:
            labeled = label_row(row)
            labeled_rows.append(labeled)
            scene_counts[str(labeled["scene_id"])] += 1
        except ValueError as exc:
            failures.append(
                {
                    "source_sample_id": row.get("source_sample_id"),
                    "error": str(exc),
                }
            )

    print(f"samples={len(rows)} labeled={len(labeled_rows)} failures={len(failures)}", flush=True)
    print("scene_counts:", dict(scene_counts), flush=True)

    if failures:
        for item in failures[:10]:
            print("failure:", item, flush=True)

    if args.dry_run:
        return 1 if failures else 0

    write_jsonl(args.output_jsonl, labeled_rows)
    print(f"output={args.output_jsonl}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
