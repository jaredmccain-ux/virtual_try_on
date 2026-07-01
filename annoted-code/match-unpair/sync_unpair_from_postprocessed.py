#!/usr/bin/env python3
"""Rebuild unpair rows from postprocessed pool, preserving existing pairings."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
CONSTRUCT_DIR = ROOT_DIR / "constrcut_instruction"
PIPELINE_DIR = CONSTRUCT_DIR / "pipeline"
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(PIPELINE_DIR))

from annotate_api import read_jsonl, write_jsonl  # noqa: E402
from build_unpair_by_scene import build_unpair_row  # noqa: E402

DEFAULT_POST = ROOT_DIR / "annotations_rivo_reviewed_supplement_checkpoint_postprocessed.jsonl"
DEFAULT_UNPAIR = SCRIPT_DIR / "annotations_rivo_unpair_200.jsonl"
DEFAULT_MANIFEST = SCRIPT_DIR / "unpair_scene_manifest.jsonl"
GARMENT_REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")


def remove_garment_is_present(annotation: dict[str, Any]) -> None:
    garment = annotation.get("garment_image", {}).get("garment_annotation")
    if not isinstance(garment, dict):
        return
    for key in GARMENT_REGION_KEYS:
        region = garment.get(key)
        if isinstance(region, dict):
            region.pop("is_present", None)


def remove_lower_fields_from_upper(region: dict[str, Any] | None) -> None:
    if not isinstance(region, dict):
        return
    ws = region.get("wearing_style")
    if isinstance(ws, dict):
        ws.pop("pants_cuff_state", None)
    ls = region.get("local_structure")
    if isinstance(ls, dict):
        ls.pop("hem_length", None)
        ls.pop("hem_shape", None)


def apply_cleanups(annotation: dict[str, Any], anchor_id: str, donor_id: str) -> None:
    remove_garment_is_present(annotation)
    if anchor_id == "upper_body_048397":
        pa = annotation.get("person_image", {}).get("person_annotation")
        if isinstance(pa, dict):
            remove_lower_fields_from_upper(pa.get("upper_body_garment"))
    if donor_id == "upper_body_048397":
        ga = annotation.get("garment_image", {}).get("garment_annotation")
        if isinstance(ga, dict):
            remove_lower_fields_from_upper(ga.get("upper_body_garment"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sync unpair JSONL from postprocessed pool.")
    parser.add_argument("--post-jsonl", type=Path, default=DEFAULT_POST)
    parser.add_argument("--unpair-jsonl", type=Path, default=DEFAULT_UNPAIR)
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool = read_jsonl(args.post_jsonl)
    pool_by_id = {str(r["source_sample_id"]): (i, r) for i, r in enumerate(pool)}
    existing = read_jsonl(args.unpair_jsonl)
    manifest_rows = read_jsonl(args.manifest_jsonl) if args.manifest_jsonl.is_file() else []

    updated: list[dict[str, Any]] = []
    updated_manifest: list[dict[str, Any]] = []

    for idx, row in enumerate(existing):
        anchor_id = str(row.get("person_anchor_id") or row.get("paired_baseline_id"))
        donor_id = str(row.get("garment_donor_id"))
        scene_id = str(row.get("scene_id"))

        if anchor_id not in pool_by_id or donor_id not in pool_by_id:
            print(f"ERROR: missing pool entry for {anchor_id} / {donor_id}", flush=True)
            return 1

        anchor_index, anchor = pool_by_id[anchor_id]
        donor_index, donor = pool_by_id[donor_id]

        new_row, manifest = build_unpair_row(
            anchor,
            donor,
            scene_id=scene_id,
            anchor_index=anchor_index,
            donor_index=donor_index,
            seed=20260623,
        )
        apply_cleanups(new_row["annotation"], anchor_id, donor_id)

        # Preserve review flags from existing row if present
        if row.get("review_edited"):
            new_row["review_edited"] = row["review_edited"]
        if row.get("reviewed_at"):
            new_row["reviewed_at"] = row["reviewed_at"]
        for key in ("supplement_meta", "postprocess_meta"):
            if key in anchor:
                new_row[key] = copy.deepcopy(anchor[key])

        updated.append(new_row)
        updated_manifest.append(manifest)

    print(f"rebuilt unpair rows: {len(updated)}", flush=True)

    if args.dry_run:
        return 0

    write_jsonl(args.unpair_jsonl, updated)
    write_jsonl(args.manifest_jsonl, updated_manifest)
    print(f"written: {args.unpair_jsonl}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
