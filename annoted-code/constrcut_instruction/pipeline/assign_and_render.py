#!/usr/bin/env python3
"""Assign E01-E20 edit types and render dimension-structured instructions."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
ANNOTED_CODE_DIR = ROOT_DIR.parent
sys.path.insert(0, str(ANNOTED_CODE_DIR))

from edit_common import (
    DIMENSIONS,
    PERSON_PRESERVE_KEYS,
    REGION_KEYS,
    REGION_FIELDS,
    attribute_label,
    build_active_region_edit_state,
    empty_changes_by_dimension,
    get_active_region,
    get_garment_annotation,
    get_person_annotation,
    humanize,
    is_valid_value,
    list_editable_attributes,
    pick_to_value,
    read_json,
    read_jsonl,
    render_layering_edit_en,
    render_layering_edit_zh,
    render_layering_preserve_en,
    render_layering_preserve_zh,
    sample_key,
    serialize_value,
    write_jsonl,
)
from edit_semantics import filter_editable_for_pair_mode, filter_editable_for_scene
from postprocess_annotations import (  # noqa: E402
    normalize_edit_attribute_value,
    process_attributes_block,
    process_garment_annotation,
    sync_edit_changes_from_attributes,
)
from scene_classify import (
    active_region_from_garment,
    adjust_preserved_for_scene,
    classify_paired_scene,
    edit_subject_prefix_en,
    edit_subject_prefix_zh,
    render_scene_base_en,
    render_scene_base_zh,
)


DEFAULT_CATALOG = PIPELINE_DIR / "edit_type_catalog.json"
DEFAULT_ENUMS = PIPELINE_DIR / "edit_value_enums.json"
DEFAULT_PAIRED_INPUT = DATA_DIR / "annotations_paired_scenes.jsonl"
DEFAULT_UNPAIR_INPUT = ANNOTED_CODE_DIR / "match-unpair" / "annotations_api_unpair_200.jsonl"
DEFAULT_PAIRED_SOURCE = (
    ANNOTED_CODE_DIR / "annotations_rivo_reviewed_supplement_checkpoint_postprocessed.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assign edit types and render instructions.")
    parser.add_argument(
        "--dataset",
        choices=["paired", "unpair", "both"],
        default="both",
        help="Which dataset(s) to process.",
    )
    parser.add_argument("--paired-input", type=Path, default=DEFAULT_PAIRED_INPUT)
    parser.add_argument("--unpair-input", type=Path, default=DEFAULT_UNPAIR_INPUT)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--enums", type=Path, default=DEFAULT_ENUMS)
    parser.add_argument("--paired-seed", type=int, default=20260621)
    parser.add_argument("--unpair-seed", type=int, default=20260622)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sample_rng(seed: int, key: str) -> random.Random:
    return random.Random(f"{seed}:{key}")


def fulfill_recipe(
    *,
    recipe: dict[str, Any],
    editable: list[tuple[str, str, Any]],
    enums: dict[str, Any],
    rng: random.Random,
) -> dict[str, dict[str, dict[str, Any]]] | None:
    available = editable[:]
    rng.shuffle(available)
    used: set[tuple[str, str]] = set()
    changes = empty_changes_by_dimension()

    for dimension, spec in recipe.items():
        need = int(spec["count"])
        allowed_attrs = set(spec["attributes"])
        picked = 0
        for dim, attribute, from_value in available:
            if picked >= need:
                break
            if dim != dimension:
                continue
            if attribute not in allowed_attrs:
                continue
            if (dim, attribute) in used:
                continue
            to_value = pick_to_value(
                dimension=dim,
                attribute=attribute,
                from_value=from_value,
                enums=enums,
                rng=rng,
            )
            if to_value is None:
                continue
            changes[dimension][attribute] = {
                "from": serialize_value(
                    normalize_edit_attribute_value(dimension, attribute, from_value)
                ),
                "to": serialize_value(to_value),
            }
            used.add((dim, attribute))
            picked += 1
        if picked < need:
            return None
    return changes


def can_fulfill_type(
    *,
    type_id: str,
    catalog: dict[str, Any],
    editable: list[tuple[str, str, Any]],
    enums: dict[str, Any],
    rng: random.Random,
) -> bool:
    recipe = catalog["types"][type_id]["recipe"]
    return (
        fulfill_recipe(recipe=recipe, editable=editable, enums=enums, rng=rng) is not None
    )


def build_changes_for_type(
    *,
    type_id: str,
    catalog: dict[str, Any],
    editable: list[tuple[str, str, Any]],
    enums: dict[str, Any],
    rng: random.Random,
) -> dict[str, dict[str, dict[str, Any]]]:
    recipe = catalog["types"][type_id]["recipe"]
    changes = fulfill_recipe(recipe=recipe, editable=editable, enums=enums, rng=rng)
    if changes is None:
        raise ValueError(f"unable to fulfill edit type {type_id}")
    return changes


def build_preserved_by_dimension(
    *,
    source_attributes: dict[str, Any],
    garment_annotation: dict[str, Any],
    active_region: str,
    changes: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    preserved: dict[str, Any] = {"person": list(PERSON_PRESERVE_KEYS)}
    edit_keys = {
        (dimension, attribute)
        for dimension, attrs in changes.items()
        for attribute in attrs
    }

    garment_baseline = build_active_region_edit_state(
        source_attributes,
        garment_annotation,
        active_region,
    )

    for region_key in REGION_KEYS:
        if region_key == active_region:
            region = garment_baseline
        else:
            region = source_attributes.get(region_key)
            if not isinstance(region, dict) or not region.get("is_present"):
                continue

        region_preserved: dict[str, dict[str, Any]] = {}
        for dimension, attributes in REGION_FIELDS.get(region_key, {}).items():
            block = region.get(dimension)
            if not isinstance(block, dict):
                continue
            kept: dict[str, Any] = {}
            for attribute in attributes:
                value = block.get(attribute)
                if not is_valid_value(value):
                    continue
                if region_key == active_region and (dimension, attribute) in edit_keys:
                    continue
                kept[attribute] = serialize_value(value)
            if kept:
                region_preserved[dimension] = kept
        if region_preserved:
            preserved[region_key] = region_preserved
    return preserved


def build_target_attributes(
    *,
    source_attributes: dict[str, Any],
    garment_annotation: dict[str, Any],
    active_region: str,
    changes: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    target = copy.deepcopy(source_attributes)
    target[active_region] = build_active_region_edit_state(
        source_attributes, garment_annotation, active_region
    )
    region = target[active_region]
    for dimension, attrs in changes.items():
        if not attrs:
            continue
        block = region.setdefault(dimension, {})
        if not isinstance(block, dict):
            block = {}
            region[dimension] = block
        for attribute, change in attrs.items():
            block[attribute] = change["to"]
    return target


def edit_prefix_for_scene(scene_id: str | None, active_region: str | None) -> tuple[str, str]:
    if not scene_id or not active_region:
        return "", ""
    if active_region == "upper_body_garment":
        return edit_subject_prefix_en(scene_id), edit_subject_prefix_zh(scene_id)
    if active_region == "whole_body_garment" and scene_id == "E":
        return edit_subject_prefix_en("E"), edit_subject_prefix_zh("E")
    return "", ""


def render_edit_sentences_en(
    changes: dict[str, dict[str, dict[str, Any]]],
    *,
    scene_id: str | None = None,
    active_region: str | None = None,
) -> list[str]:
    prefix, _ = edit_prefix_for_scene(scene_id, active_region)

    sentences: list[str] = []
    for dimension in DIMENSIONS:
        attrs = changes.get(dimension) or {}
        if not attrs:
            continue
        if dimension == "layering_structure":
            parts = [
                render_layering_edit_en(
                    attribute,
                    change["from"],
                    change["to"],
                    scene_id=scene_id,
                )
                for attribute, change in attrs.items()
            ]
        else:
            parts = [
                f"change the {attribute_label(attribute)} from {humanize(change['from'])} "
                f"to {humanize(change['to'])}"
                for attribute, change in attrs.items()
            ]
        joined = "; ".join(parts)
        body = f"[EDIT · {dimension}] {prefix}{joined.capitalize()}."
        sentences.append(body)
    return sentences


def render_edit_sentences_zh(
    changes: dict[str, dict[str, dict[str, Any]]],
    *,
    scene_id: str | None = None,
    active_region: str | None = None,
) -> list[str]:
    _, prefix = edit_prefix_for_scene(scene_id, active_region)

    sentences: list[str] = []
    for dimension in DIMENSIONS:
        attrs = changes.get(dimension) or {}
        if not attrs:
            continue
        if dimension == "layering_structure":
            parts = [
                render_layering_edit_zh(
                    attribute,
                    change["from"],
                    change["to"],
                    scene_id=scene_id,
                )
                for attribute, change in attrs.items()
            ]
        else:
            parts = [
                f"将{attribute_label(attribute)}从 {humanize(change['from'])} 改为 {humanize(change['to'])}"
                for attribute, change in attrs.items()
            ]
        sentences.append(f"[编辑 · {dimension}] {prefix}" + "；".join(parts) + "。")
    return sentences


def render_preserve_sentences_en(preserved: dict[str, Any]) -> list[str]:
    sentences: list[str] = []
    sentences.append(
        "[PRESERVE · person] Preserve identity, face, pose, body shape, background, and lighting."
    )
    for region_key in REGION_KEYS:
        region_block = preserved.get(region_key)
        if not isinstance(region_block, dict):
            continue
        chunks: list[str] = []
        for dimension in DIMENSIONS:
            attrs = region_block.get(dimension)
            if not isinstance(attrs, dict):
                continue
            if dimension == "layering_structure":
                layering_phrase = render_layering_preserve_en(attrs)
                if layering_phrase:
                    chunks.append(layering_phrase)
                continue
            for attribute in attrs:
                chunks.append(f"{attribute_label(attribute)} unchanged")
        if chunks:
            label = region_key.replace("_garment", "")
            sentences.append(
                f"[PRESERVE · {label}] Keep " + ", ".join(chunks) + "."
            )
    return sentences


def render_preserve_sentences_zh(preserved: dict[str, Any]) -> list[str]:
    sentences: list[str] = []
    sentences.append(
        "[保留 · person] 保持身份、面部、姿态、体型、背景和光照不变。"
    )
    for region_key in REGION_KEYS:
        region_block = preserved.get(region_key)
        if not isinstance(region_block, dict):
            continue
        chunks: list[str] = []
        for dimension in DIMENSIONS:
            attrs = region_block.get(dimension)
            if not isinstance(attrs, dict):
                continue
            if dimension == "layering_structure":
                layering_phrase = render_layering_preserve_zh(attrs)
                if layering_phrase:
                    chunks.append(layering_phrase)
                continue
            for attribute in attrs:
                chunks.append(f"{attribute_label(attribute)} 不变")
        if chunks:
            label = region_key.replace("_garment", "")
            sentences.append(
                f"[保留 · {label}] 保持 " + "、".join(chunks) + "。"
            )
    return sentences


def resolve_scene_id(row: dict[str, Any]) -> str:
    scene_id = row.get("scene_id")
    if isinstance(scene_id, str) and scene_id:
        return scene_id
    person_annotation = get_person_annotation(row)
    garment_annotation = get_garment_annotation(row)
    classified = classify_paired_scene(person_annotation, garment_annotation)
    if classified is None:
        raise ValueError(f"{sample_key(row)}: missing scene_id and unable to classify")
    return classified


def render_instruction(
    *,
    changes: dict[str, dict[str, dict[str, Any]]],
    preserved: dict[str, Any],
    scene_id: str,
    garment_annotation: dict[str, Any],
    active_region: str,
    source_attributes: dict[str, Any],
    pair_mode: str | None = None,
) -> tuple[str, str, dict[str, Any]]:
    preserved = adjust_preserved_for_scene(
        preserved,
        scene_id=scene_id,
        source_attributes=source_attributes,
        active_region=active_region,
        pair_mode=pair_mode,
    )
    spec = {
        "scene_id": scene_id,
        "base_task": scene_id,
        "edit_by_dimension": changes,
        "preserve_by_dimension": preserved,
    }
    en_prefix, zh_prefix = edit_prefix_for_scene(scene_id, active_region)
    transition_needed = not (en_prefix or zh_prefix)
    en_parts = [render_scene_base_en(scene_id, garment_annotation)]
    if transition_needed:
        en_parts.append("After putting on the given garment,")
    en_parts.extend(
        render_edit_sentences_en(
            changes, scene_id=scene_id, active_region=active_region
        )
    )
    en_parts.extend(render_preserve_sentences_en(preserved))
    zh_parts = [render_scene_base_zh(scene_id, garment_annotation)]
    if transition_needed:
        zh_parts.append("在换上给定的衣物后，")
    zh_parts.extend(
        render_edit_sentences_zh(
            changes, scene_id=scene_id, active_region=active_region
        )
    )
    zh_parts.extend(render_preserve_sentences_zh(preserved))
    return " ".join(en_parts), "".join(zh_parts), spec


def type_difficulty(catalog: dict[str, Any], type_id: str) -> int:
    recipe = catalog["types"][type_id]["recipe"]
    return sum(int(spec["count"]) for spec in recipe.values())


def sort_types_for_assignment(catalog: dict[str, Any], rng: random.Random) -> list[str]:
    type_ids = list(catalog["types"].keys())
    type_ids.sort(
        key=lambda type_id: (
            -type_difficulty(catalog, type_id),
            rng.random(),
        )
    )
    return type_ids


def get_type_target(catalog: dict[str, Any], type_id: str) -> int:
    type_def = catalog["types"][type_id]
    return int(type_def.get("target_count", catalog.get("default_target_count", 10)))


def infer_pair_mode(row: dict[str, Any]) -> str:
    return str(row.get("pair_mode") or "paired")


def reshape_edit_type_quotas(
    assignments: dict[int, str],
    counts: Counter[str],
    *,
    prepared: dict[int, dict[str, Any]],
    fulfillable_by_index: dict[int, list[str]],
    catalog: dict[str, Any],
    enums: dict[str, Any],
    rng: random.Random,
) -> None:
    """Re-home samples after quota changes (e.g. E09/E10 15->10, new E08=10)."""
    type_ids = list(catalog["types"].keys())
    e08_target = get_type_target(catalog, "E08")

    def type_count(type_id: str) -> int:
        return sum(1 for assigned in assignments.values() if assigned == type_id)

    def can_assign(index: int, type_id: str) -> bool:
        if type_id not in fulfillable_by_index.get(index, []):
            return False
        info = prepared[index]
        return can_fulfill_type(
            type_id=type_id,
            catalog=catalog,
            editable=info["editable"],
            enums=enums,
            rng=info["rng"],
        )

    def reassign_index(index: int, *, prefer: list[str]) -> None:
        under = [
            type_id
            for type_id in type_ids
            if type_count(type_id) < get_type_target(catalog, type_id)
        ]
        under.sort(
            key=lambda type_id: get_type_target(catalog, type_id) - type_count(type_id),
            reverse=True,
        )
        for new_type in prefer + under + type_ids:
            if not can_assign(index, new_type):
                continue
            if new_type in prefer or type_count(new_type) < get_type_target(catalog, new_type):
                assignments[index] = new_type
                return
        for new_type in type_ids:
            if can_assign(index, new_type):
                assignments[index] = new_type
                return

    reassign_queue: list[int] = []
    for trim_type in ("E09", "E10"):
        target = get_type_target(catalog, trim_type)
        indices = [index for index, type_id in assignments.items() if type_id == trim_type]
        rng.shuffle(indices)
        reassign_queue.extend(indices[target:])

    e08_indices = [index for index, type_id in assignments.items() if type_id == "E08"]
    rng.shuffle(e08_indices)
    reassign_queue.extend(e08_indices[e08_target:])

    rng.shuffle(reassign_queue)
    deferred: list[int] = []
    for index in reassign_queue:
        if type_count("E08") < e08_target and can_assign(index, "E08"):
            assignments[index] = "E08"
        else:
            deferred.append(index)

    for index in deferred:
        reassign_index(index, prefer=[])

    for type_id in type_ids:
        counts[type_id] = type_count(type_id)


def enforce_type_caps(
    assignments: dict[int, str],
    counts: Counter[str],
    *,
    prepared: dict[int, dict[str, Any]],
    fulfillable_by_index: dict[int, list[str]],
    catalog: dict[str, Any],
    enums: dict[str, Any],
    rng: random.Random,
) -> None:
    """Trim over-target edit types by reassigning samples to under-target ones."""
    type_ids = list(catalog["types"].keys())

    def under_target() -> list[str]:
        return [
            type_id
            for type_id in type_ids
            if counts[type_id] < get_type_target(catalog, type_id)
        ]

    changed = True
    while changed:
        changed = False
        moves: list[tuple[int, str]] = []
        for type_id in type_ids:
            target = get_type_target(catalog, type_id)
            excess = counts[type_id] - target
            if excess <= 0:
                continue
            indices = [index for index, tid in assignments.items() if tid == type_id]
            rng.shuffle(indices)
            moves.extend((index, type_id) for index in indices[:excess])

        rng.shuffle(moves)
        for index, old_type in moves:
            if counts[old_type] <= get_type_target(catalog, old_type):
                continue
            candidates = under_target()
            candidates.sort(
                key=lambda type_id: get_type_target(catalog, type_id) - counts[type_id],
                reverse=True,
            )
            for new_type in candidates:
                if new_type not in fulfillable_by_index.get(index, []):
                    continue
                info = prepared[index]
                if not can_fulfill_type(
                    type_id=new_type,
                    catalog=catalog,
                    editable=info["editable"],
                    enums=enums,
                    rng=info["rng"],
                ):
                    continue
                assignments[index] = new_type
                counts[old_type] -= 1
                counts[new_type] += 1
                changed = True
                break


def assign_edit_types(
    rows: list[dict[str, Any]],
    *,
    catalog: dict[str, Any],
    enums: dict[str, Any],
    seed: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    type_ids = list(catalog["types"].keys())
    counts = Counter({type_id: 0 for type_id in type_ids})
    failures: list[dict[str, Any]] = []
    assignments: dict[int, str] = {}

    indexed_rows = list(enumerate(rows))
    unassigned = [index for index, _ in indexed_rows]
    rng_order = random.Random(seed)
    rng_order.shuffle(unassigned)
    type_order = sort_types_for_assignment(catalog, rng_order)

    prepared: dict[int, dict[str, Any]] = {}
    for index, row in indexed_rows:
        key = sample_key(row)
        source_attributes = get_person_annotation(row)
        process_attributes_block(source_attributes)
        garment_annotation = get_garment_annotation(row)
        process_garment_annotation(garment_annotation)
        active_region = get_active_region(row)
        scene_id = resolve_scene_id(row)
        pair_mode = infer_pair_mode(row)
        editable = list_editable_attributes(
            source_attributes, garment_annotation, active_region
        )
        editable = filter_editable_for_scene(
            editable,
            scene_id=scene_id,
            person_annotation=source_attributes,
            enums=enums,
        )
        editable = filter_editable_for_pair_mode(
            editable,
            pair_mode=pair_mode,
        )
        prepared[index] = {
            "key": key,
            "source_attributes": source_attributes,
            "garment_annotation": garment_annotation,
            "active_region": active_region,
            "scene_id": scene_id,
            "pair_mode": pair_mode,
            "editable": editable,
            "rng": sample_rng(seed, key),
        }

    fulfillable_by_index: dict[int, list[str]] = {}
    for index in prepared:
        info = prepared[index]
        fulfillable_by_index[index] = [
            type_id
            for type_id in type_ids
            if can_fulfill_type(
                type_id=type_id,
                catalog=catalog,
                editable=info["editable"],
                enums=enums,
                rng=info["rng"],
            )
        ]

    def try_assign(index: int, type_id: str, *, enforce_cap: bool) -> bool:
        if index in assignments:
            return False
        if enforce_cap and counts[type_id] >= get_type_target(catalog, type_id):
            return False
        info = prepared[index]
        if not can_fulfill_type(
            type_id=type_id,
            catalog=catalog,
            editable=info["editable"],
            enums=enums,
            rng=info["rng"],
        ):
            return False
        assignments[index] = type_id
        counts[type_id] += 1
        return True

    # Phase 1: iteratively fill the most under-filled feasible edit type.
    while True:
        deficits = [
            (get_type_target(catalog, type_id) - counts[type_id], type_id)
            for type_id in type_order
            if counts[type_id] < get_type_target(catalog, type_id)
        ]
        if not deficits or not unassigned:
            break
        deficits.sort(reverse=True)
        assigned_this_round = False
        for _, type_id in deficits:
            candidates = [
                index
                for index in unassigned
                if type_id in fulfillable_by_index.get(index, [])
            ]
            if not candidates:
                continue
            candidates.sort(
                key=lambda index: (
                    len(fulfillable_by_index.get(index, [])),
                    rng_order.random(),
                )
            )
            index = candidates[0]
            if try_assign(index, type_id, enforce_cap=True):
                unassigned.remove(index)
                assigned_this_round = True
                break
        if not assigned_this_round:
            break

    # Phase 2: assign remaining samples to any fulfillable type (fallback aware).
    for index in unassigned[:]:
        if index in assignments:
            continue
        info = prepared[index]

        def pick_type(candidates: list[str], *, prefer_under_target: bool) -> str | None:
            if not candidates:
                return None
            fulfillable = fulfillable_by_index.get(index, [])
            under = [
                type_id
                for type_id in candidates
                if counts[type_id] < get_type_target(catalog, type_id)
            ]
            search_order: list[list[str]] = []
            if prefer_under_target and under:
                search_order.append(under)
            search_order.append(candidates)
            for pool in search_order:
                filtered = [type_id for type_id in pool if type_id in fulfillable]
                if filtered:
                    filtered.sort(key=lambda type_id: (counts[type_id], rng_order.random()))
                    return filtered[0]
            return None

        chosen = None
        open_types = [
            type_id
            for type_id in type_ids
            if counts[type_id] < get_type_target(catalog, type_id)
        ]
        rng_order.shuffle(open_types)
        picked = pick_type(open_types, prefer_under_target=True)
        if picked and try_assign(index, picked, enforce_cap=True):
            chosen = picked
            unassigned.remove(index)

        if chosen is not None:
            continue

        fallback_candidates: list[str] = []
        for type_id in type_ids:
            fallback = catalog["types"][type_id].get("fallback")
            if fallback:
                fallback_candidates.append(str(fallback))
        picked = pick_type(list(type_ids) + fallback_candidates, prefer_under_target=False)
        if picked and try_assign(index, picked, enforce_cap=False):
            unassigned.remove(index)
            chosen = picked

        if chosen is None:
            failures.append(
                {
                    "index": index,
                    "sample_key": info["key"],
                    "active_region": info["active_region"],
                    "scene_id": info.get("scene_id"),
                    "reason": "no_fulfillable_edit_type",
                }
            )

    reshape_edit_type_quotas(
        assignments,
        counts,
        prepared=prepared,
        fulfillable_by_index=fulfillable_by_index,
        catalog=catalog,
        enums=enums,
        rng=rng_order,
    )

    enforce_type_caps(
        assignments,
        counts,
        prepared=prepared,
        fulfillable_by_index=fulfillable_by_index,
        catalog=catalog,
        enums=enums,
        rng=rng_order,
    )

    output_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        if index not in assignments:
            output_rows.append(copy.deepcopy(row))
            continue

        type_id = assignments[index]
        info = prepared[index]
        source_attributes = info["source_attributes"]
        active_region = info["active_region"]
        changes = build_changes_for_type(
            type_id=type_id,
            catalog=catalog,
            editable=info["editable"],
            enums=enums,
            rng=info["rng"],
        )
        preserved = build_preserved_by_dimension(
            source_attributes=source_attributes,
            garment_annotation=info["garment_annotation"],
            active_region=active_region,
            changes=changes,
        )
        baseline_attributes = build_target_attributes(
            source_attributes=source_attributes,
            garment_annotation=info["garment_annotation"],
            active_region=active_region,
            changes={},
        )
        process_attributes_block(baseline_attributes)
        target_attributes = build_target_attributes(
            source_attributes=source_attributes,
            garment_annotation=info["garment_annotation"],
            active_region=active_region,
            changes=changes,
        )
        process_attributes_block(target_attributes)
        sync_edit_changes_from_attributes(
            changes,
            source_attributes=baseline_attributes,
            target_attributes=target_attributes,
            active_region=active_region,
        )
        for dimension, attrs in changes.items():
            if not attrs:
                continue
            for attribute, change in attrs.items():
                change["from"] = serialize_value(change.get("from"))
                change["to"] = serialize_value(change.get("to"))

        scene_id = info.get("scene_id") or resolve_scene_id(row)
        instruction_en, instruction_zh, instruction_spec = render_instruction(
            changes=changes,
            preserved=preserved,
            scene_id=scene_id,
            garment_annotation=info["garment_annotation"],
            active_region=active_region,
            source_attributes=source_attributes,
            pair_mode=info.get("pair_mode"),
        )
        preserved = instruction_spec["preserve_by_dimension"]

        enriched = copy.deepcopy(row)
        enriched["pair_mode"] = infer_pair_mode(row)
        enriched["scene_id"] = scene_id
        if not enriched.get("active_region"):
            enriched["active_region"] = active_region
        enriched["edit_task"] = {
            "edit_type_id": type_id,
            "edit_type_label": catalog["types"][type_id]["label"],
            "scene_id": scene_id,
            "region": active_region,
            "edit_spec": changes,
            "changes_by_dimension": changes,
            "preserved_by_dimension": preserved,
        }
        enriched["source_attributes"] = copy.deepcopy(source_attributes)
        enriched["target_attributes"] = target_attributes
        enriched["instruction"] = {
            "instruction_spec": instruction_spec,
            "instruction_en": instruction_en,
            "instruction_zh": instruction_zh,
            "template_version": "edit_instruction_v3_scene",
        }
        if not enriched.get("sample_id"):
            enriched["sample_id"] = info["key"]
        output_rows.append(enriched)
        assignment_rows.append(
            {
                "sample_key": info["key"],
                "scene_id": scene_id,
                "edit_type_id": type_id,
                "edit_type_label": catalog["types"][type_id]["label"],
                "active_region": active_region,
                "instruction_en": instruction_en,
            }
        )

    summary = [
        {
            "edit_type_id": type_id,
            "count": counts[type_id],
            "target_count": get_type_target(catalog, type_id),
            "eligible_failures": max(
                0, get_type_target(catalog, type_id) - counts[type_id]
            ),
        }
        for type_id in type_ids
    ]
    return output_rows, assignment_rows, failures, summary


def write_summary_csv(
    path: Path, summary: list[dict[str, Any]], pair_mode: str
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "edit_type_id",
                "pair_mode",
                "count",
                "target_count",
                "eligible_failures",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow(
                {
                    "edit_type_id": row["edit_type_id"],
                    "pair_mode": pair_mode,
                    "count": row["count"],
                    "target_count": row["target_count"],
                    "eligible_failures": row["eligible_failures"],
                }
            )


def process_dataset(
    *,
    pair_mode: str,
    input_path: Path,
    output_path: Path,
    assignment_path: Path,
    summary_path: Path,
    failures_path: Path,
    catalog: dict[str, Any],
    enums: dict[str, Any],
    seed: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"input not found: {input_path}")

    for path in (output_path, assignment_path, summary_path, failures_path):
        if path.exists() and not overwrite and not dry_run:
            raise FileExistsError(f"{path} exists; pass --overwrite")

    rows = read_jsonl(input_path)
    output_rows, assignment_rows, failures, summary = assign_edit_types(
        rows, catalog=catalog, enums=enums, seed=seed
    )

    assigned = len(assignment_rows)
    print(f"\n=== {pair_mode} ===")
    print(f"input={input_path}")
    print(f"samples={len(rows)} assigned={assigned} failures={len(failures)}")
    print("edit_type counts:")
    for row in summary:
        print(
            f"  {row['edit_type_id']}: {row['count']}/{row['target_count']}"
            f" (eligible_failures={row['eligible_failures']})"
        )

    if dry_run:
        if assignment_rows:
            sample = assignment_rows[0]
            print("\nexample instruction:")
            print(sample["instruction_en"][:500] + "...")
        return

    write_jsonl(output_path, output_rows)
    write_jsonl(assignment_path, assignment_rows)
    write_jsonl(failures_path, failures)
    write_summary_csv(summary_path, summary, pair_mode)
    print(f"output={output_path}")
    print(f"assignments={assignment_path}")
    print(f"summary={summary_path}")
    print(f"failures={failures_path}")


def main() -> int:
    args = parse_args()
    catalog = read_json(args.catalog)
    enums = read_json(args.enums)

    if args.dataset in ("paired", "both"):
        process_dataset(
            pair_mode="paired",
            input_path=args.paired_input,
            output_path=DATA_DIR / "annotations_paired_with_edit.jsonl",
            assignment_path=DATA_DIR / "paired_edit_assignments.jsonl",
            summary_path=DATA_DIR / "paired_edit_type_summary.csv",
            failures_path=DATA_DIR / "paired_edit_failures.jsonl",
            catalog=catalog,
            enums=enums,
            seed=args.paired_seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    if args.dataset in ("unpair", "both"):
        process_dataset(
            pair_mode="unpair",
            input_path=args.unpair_input,
            output_path=DATA_DIR / "annotations_unpair_with_edit.jsonl",
            assignment_path=DATA_DIR / "unpair_edit_assignments.jsonl",
            summary_path=DATA_DIR / "unpair_edit_type_summary.csv",
            failures_path=DATA_DIR / "unpair_edit_failures.jsonl",
            catalog=catalog,
            enums=enums,
            seed=args.unpair_seed,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
