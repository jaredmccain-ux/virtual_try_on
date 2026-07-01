#!/usr/bin/env python3
"""Sync annotation from postprocessed checkpoint into annotations_paired_scenes.jsonl."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
ANNOTED_CODE_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ANNOTED_CODE_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from annotate_api import read_jsonl, write_jsonl  # noqa: E402
from label_paired_scenes import label_row  # noqa: E402

DEFAULT_POST = ANNOTED_CODE_DIR / "annotations_rivo_reviewed_supplement_checkpoint_postprocessed.jsonl"
DEFAULT_SCENES = DATA_DIR / "annotations_paired_scenes.jsonl"
GARMENT_REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")
SCENE_FIELDS = ("scene_id", "person_upper_state", "garment_class", "active_region", "source_attributes")
META_FIELDS = ("supplement_meta", "postprocess_meta")


def remove_garment_is_present(annotation: dict[str, Any]) -> int:
    removed = 0
    garment = annotation.get("garment_image", {}).get("garment_annotation")
    if not isinstance(garment, dict):
        return removed
    for key in GARMENT_REGION_KEYS:
        region = garment.get(key)
        if isinstance(region, dict) and "is_present" in region:
            del region["is_present"]
            removed += 1
    return removed


def remove_lower_fields_from_upper(region: dict[str, Any] | None) -> None:
    if not isinstance(region, dict):
        return
    wearing_style = region.get("wearing_style")
    if isinstance(wearing_style, dict):
        wearing_style.pop("pants_cuff_state", None)
    local_structure = region.get("local_structure")
    if isinstance(local_structure, dict):
        local_structure.pop("hem_length", None)
        local_structure.pop("hem_shape", None)


def apply_cleanups(annotation: dict[str, Any], source_sample_id: str) -> int:
    removed = remove_garment_is_present(annotation)
    if source_sample_id == "upper_body_048397":
        person_annotation = annotation.get("person_image", {}).get("person_annotation")
        if isinstance(person_annotation, dict):
            remove_lower_fields_from_upper(person_annotation.get("upper_body_garment"))
        garment = annotation.get("garment_image", {}).get("garment_annotation")
        if isinstance(garment, dict):
            remove_lower_fields_from_upper(garment.get("upper_body_garment"))
    return removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync postprocessed annotations into annotations_paired_scenes.jsonl."
    )
    parser.add_argument("--post-jsonl", type=Path, default=DEFAULT_POST)
    parser.add_argument("--scenes-jsonl", type=Path, default=DEFAULT_SCENES)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    scenes_rows = read_jsonl(args.scenes_jsonl)
    post_rows = read_jsonl(args.post_jsonl)
    post_by_id = {str(row["source_sample_id"]): row for row in post_rows}

    missing_in_post = [row["source_sample_id"] for row in scenes_rows if row["source_sample_id"] not in post_by_id]
    if missing_in_post:
        print(f"ERROR: {len(missing_in_post)} samples missing in postprocessed", flush=True)
        return 1

    garment_is_present_removed = 0
    scene_changes: list[tuple[str, str, str]] = []
    updated_rows: list[dict[str, Any]] = []

    for row in scenes_rows:
        sid = str(row["source_sample_id"])
        merged = copy.deepcopy(row)
        post_row = post_by_id[sid]

        merged["annotation"] = copy.deepcopy(post_row["annotation"])
        garment_is_present_removed += apply_cleanups(merged["annotation"], sid)

        for key in META_FIELDS:
            if key in post_row:
                merged[key] = copy.deepcopy(post_row[key])

        # Drop stale source_attributes so label_row reads synced person_annotation.
        merged.pop("source_attributes", None)
        labeled = label_row(merged)
        old_scene = merged.get("scene_id")
        for key in SCENE_FIELDS:
            merged[key] = labeled[key]
        if old_scene != merged.get("scene_id"):
            scene_changes.append((sid, str(old_scene), str(merged.get("scene_id"))))

        updated_rows.append(merged)

    print(f"scenes rows: {len(scenes_rows)}", flush=True)
    print(f"garment is_present removed (count): {garment_is_present_removed}", flush=True)
    print(f"scene_id changes: {len(scene_changes)}", flush=True)
    for sid, old, new in scene_changes:
        print(f"  {sid}: {old} -> {new}", flush=True)

    if args.dry_run:
        print("dry-run: no file written", flush=True)
        return 0

    write_jsonl(args.scenes_jsonl, updated_rows)
    print(f"written: {args.scenes_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
