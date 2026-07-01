#!/usr/bin/env python3
"""Build unpair dataset with scene quotas (gender gate + scene eligibility)."""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from annotate_api import read_jsonl, write_jsonl  # noqa: E402
from constrcut_instruction.pipeline.scene_classify import (  # noqa: E402
    DEFAULT_SCENE_QUOTAS,
    UNPAIR_SCENE_ASSIGN_ORDER,
    active_region_from_garment,
    eligible_scenes,
    gender_compatible,
    garment_target,
    person_wearing_type,
)

DEFAULT_INPUT = SCRIPT_DIR.parent / "annotations_rivo_reviewed_supplement_checkpoint_postprocessed.jsonl"
DEFAULT_OUTPUT = SCRIPT_DIR / "annotations_rivo_unpair_200.jsonl"
DEFAULT_MANIFEST = SCRIPT_DIR / "unpair_scene_manifest.jsonl"

WHERE_TO_REGION = {
    "upper_body": "upper_body_garment",
    "lower_body": "lower_body_garment",
    "whole_body": "whole_body_garment",
}
REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scene-stratified unpair JSONL.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-jsonl", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--seed", type=int, default=20260623)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def extract_annotation(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    annotation = row["annotation"]
    person_image = copy.deepcopy(annotation["person_image"])
    garment_image = copy.deepcopy(annotation["garment_image"])
    return person_image, garment_image


def person_role(row: dict[str, Any]) -> str | None:
    role = row.get("annotation", {}).get("person_image", {}).get("role")
    return str(role) if role else None


def garment_gender(row: dict[str, Any]) -> str | None:
    gender = (
        row.get("annotation", {})
        .get("garment_image", {})
        .get("garment_annotation", {})
        .get("gender")
    )
    return str(gender) if gender else None


def recompute_is_paired(person_annotation: dict[str, Any], active_region: str) -> None:
    for region_key in REGION_KEYS:
        region = person_annotation.get(region_key)
        if not isinstance(region, dict):
            continue
        if not region.get("is_present"):
            continue
        region["is_paired"] = region_key == active_region


def anchor_scene_union(
    anchor_row: dict[str, Any],
    pool: list[dict[str, Any]],
    anchor_index: int,
) -> dict[str, list[int]]:
    """Map scene_id -> donor indices (gender-compatible, scene-eligible)."""
    person_image, _ = extract_annotation(anchor_row)
    person_annotation = person_image["person_annotation"]
    anchor_role = person_role(anchor_row)
    scene_donors: dict[str, list[int]] = {scene: [] for scene in DEFAULT_SCENE_QUOTAS}

    for donor_index, donor_row in enumerate(pool):
        if donor_index == anchor_index:
            continue
        if not gender_compatible(
            person_role=anchor_role,
            garment_gender=garment_gender(donor_row),
        ):
            continue
        garment_annotation = donor_row["annotation"]["garment_image"]["garment_annotation"]
        scenes = eligible_scenes(person_annotation, garment_annotation)
        scenes.discard("F")
        for scene_id in scenes:
            if scene_id in scene_donors:
                scene_donors[scene_id].append(donor_index)
    return scene_donors


def build_unpair_row(
    anchor: dict[str, Any],
    donor: dict[str, Any],
    *,
    scene_id: str,
    anchor_index: int,
    donor_index: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    person_anchor_id = str(anchor["source_sample_id"])
    garment_donor_id = str(donor["source_sample_id"])

    person_image, _ = extract_annotation(anchor)
    _, garment_image = extract_annotation(donor)
    person_annotation = person_image["person_annotation"]
    garment_annotation = garment_image["garment_annotation"]

    active_region = active_region_from_garment(garment_annotation)
    recompute_is_paired(person_annotation, active_region)

    wear_kind, upper_state = person_wearing_type(person_annotation)
    _, garment_class, _ = garment_target(person_annotation, garment_annotation)
    anchor_where = anchor["annotation"]["garment_image"]["garment_annotation"]["where_to_dress"]
    donor_where = garment_annotation["where_to_dress"]

    sample_id = f"unpair_{person_anchor_id}__garment_{garment_donor_id}"

    row = {
        "sample_id": sample_id,
        "pair_mode": "unpair",
        "scene_id": scene_id,
        "person_upper_state": upper_state or wear_kind,
        "garment_class": garment_class,
        "person_anchor_id": person_anchor_id,
        "garment_donor_id": garment_donor_id,
        "paired_baseline_id": person_anchor_id,
        "provider": anchor.get("provider", donor.get("provider")),
        "model_id": anchor.get("model_id", donor.get("model_id")),
        "model": anchor.get("model", donor.get("model")),
        "person_image": anchor.get("person_image"),
        "garment_image": donor.get("garment_image"),
        "raw_output_file": anchor.get("raw_output_file"),
        "annotation": {
            "person_image": person_image,
            "garment_image": garment_image,
        },
        "status": "ok",
        "review_verdict": anchor.get("review_verdict", "ok"),
        "source_attributes": copy.deepcopy(person_annotation),
        "active_region": active_region,
    }

    manifest = {
        "sample_id": sample_id,
        "scene_id": scene_id,
        "pair_mode": "unpair",
        "seed": seed,
        "anchor_index": anchor_index,
        "donor_index": donor_index,
        "person_anchor_id": person_anchor_id,
        "garment_donor_id": garment_donor_id,
        "person_upper_state": upper_state or wear_kind,
        "garment_class": garment_class,
        "anchor_where_to_dress": anchor_where,
        "donor_where_to_dress": donor_where,
        "active_region": active_region,
        "cross_region": anchor_where != donor_where,
    }
    return row, manifest


def assign_unpair_scenes(
    pool: list[dict[str, Any]],
    *,
    quotas: dict[str, int],
    seed: int,
) -> tuple[list[tuple[int, int, str]], dict[str, int], list[str]]:
    rng = random.Random(seed)
    remaining = {scene: count for scene, count in quotas.items() if count > 0}
    assignments: list[tuple[int, int, str]] = []
    errors: list[str] = []

    anchor_maps: list[dict[str, list[int]]] = [
        anchor_scene_union(row, pool, index) for index, row in enumerate(pool)
    ]

    assigned_anchors: set[int] = set()

    def anchor_priority(index: int) -> tuple[int, int]:
        rare_scenes = ("L2", "I2", "I1", "L1", "L3", "E")
        rare_count = sum(
            1
            for scene_id in rare_scenes
            if anchor_maps[index].get(scene_id)
        )
        return (-rare_count, index)

    def assign_scene(scene_id: str) -> None:
        need = remaining.get(scene_id, 0)
        if need <= 0:
            return
        candidates = [
            index
            for index in range(len(pool))
            if index not in assigned_anchors and anchor_maps[index].get(scene_id)
        ]
        candidates.sort(key=anchor_priority)
        rng.shuffle(candidates)

        for anchor_index in candidates:
            if need <= 0:
                break
            donor_indices = anchor_maps[anchor_index][scene_id]
            if not donor_indices:
                continue
            donor_index = rng.choice(donor_indices)
            assignments.append((anchor_index, donor_index, scene_id))
            assigned_anchors.add(anchor_index)
            need -= 1

        remaining[scene_id] = need
        if need > 0:
            errors.append(f"scene {scene_id} short by {need}")

    for scene_id in UNPAIR_SCENE_ASSIGN_ORDER:
        if scene_id in remaining:
            assign_scene(scene_id)

    overflow_order = ("A", "B", "C", "D", "L1", "L3", "E", "L2", "I2", "I1")
    for anchor_index in range(len(pool)):
        if anchor_index in assigned_anchors:
            continue
        for scene_id in overflow_order:
            donor_indices = anchor_maps[anchor_index].get(scene_id) or []
            if not donor_indices:
                continue
            donor_index = rng.choice(donor_indices)
            assignments.append((anchor_index, donor_index, scene_id))
            assigned_anchors.add(anchor_index)
            if scene_id in remaining and remaining[scene_id] > 0:
                remaining[scene_id] -= 1
            break
        else:
            errors.append(f"anchor {anchor_index} has no eligible scene")

    scene_counts = {scene: 0 for scene in quotas}
    for _, _, scene_id in assignments:
        scene_counts[scene_id] += 1

    return assignments, scene_counts, errors


def main() -> int:
    args = parse_args()
    if not args.input_jsonl.is_file():
        raise FileNotFoundError(f"input not found: {args.input_jsonl}")

    for path in (args.output_jsonl, args.manifest_jsonl):
        if path.exists() and not args.overwrite and not args.dry_run:
            raise FileExistsError(f"{path} exists; pass --overwrite")

    pool = read_jsonl(args.input_jsonl)
    if len(pool) != sum(DEFAULT_SCENE_QUOTAS.values()):
        print(
            f"warning: pool size={len(pool)} quota total={sum(DEFAULT_SCENE_QUOTAS.values())}",
            flush=True,
        )

    assignments, scene_counts, errors = assign_unpair_scenes(
        pool,
        quotas=DEFAULT_SCENE_QUOTAS,
        seed=args.seed,
    )

    print(f"anchors={len(pool)} assignments={len(assignments)}", flush=True)
    print("scene_counts:", scene_counts, flush=True)
    if errors:
        print("errors:", errors, flush=True)
    if sum(scene_counts.values()) != len(pool):
        errors.append(
            f"assignment count mismatch: {sum(scene_counts.values())} != {len(pool)}"
        )

    if args.dry_run:
        print("dry_run=true", flush=True)
        return 1 if errors else 0

    unpair_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    for anchor_index, donor_index, scene_id in assignments:
        row, manifest = build_unpair_row(
            pool[anchor_index],
            pool[donor_index],
            scene_id=scene_id,
            anchor_index=anchor_index,
            donor_index=donor_index,
            seed=args.seed,
        )
        unpair_rows.append(row)
        manifest_rows.append(manifest)

    write_jsonl(args.output_jsonl, unpair_rows)
    write_jsonl(args.manifest_jsonl, manifest_rows)
    print(f"output={args.output_jsonl}", flush=True)
    print(f"manifest={args.manifest_jsonl}", flush=True)
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
