#!/usr/bin/env python3
"""Deterministic post-processing for reviewed/supplement annotation JSONL rows."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from annotate_api import (
    PROJECT_ROOT,
    RIVO_DEFAULT_BASE_URL,
    RIVO_DEFAULT_MODEL,
    call_openai_chat_completions,
    extract_json_object,
    image_path_to_data_url,
    read_jsonl,
    write_jsonl,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "annotations_rivo_reviewed_supplement_checkpoint.jsonl"
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Datasets" / "eval_firsttest"

REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")
LAYERING_REGION_KEYS = ("upper_body_garment", "whole_body_garment")
OUTERWEAR_CATEGORIES = frozenset({"cardigan", "coat", "jacket"})
SHIRT_NECKLINE_NA_CATEGORIES = frozenset({"shirt"})

ROUND_NECK_ALIASES = frozenset(
    {"round", "round_neck", "crew_neck", "round neck", "crew neck"}
)
HALF_TUCK_ALIASES = frozenset(
    {"half_tuck", "half tuck", "french_tucked", "french_tuck", "french tuck"}
)

LAYERING_RELABEL_SYSTEM_PROMPT = """You are a fashion image analyst. You will see ONE image: a model/person photo.

The person is wearing an outerwear upper-body garment (cardigan, coat, or jacket).
is_outerwear has already been confirmed as true. Do NOT change is_outerwear.

Your task: judge ONLY how this outer garment is worn in terms of closure and layering.

## outer_closure — how the outer garment is fastened / worn on the body

- open: the outer garment is worn open (front placket, zipper, or buttons not fully closed; front opening visible).
- closed: the outer garment is worn closed (zipper/buttons/fastening done up, or naturally closed such as a pullover-style cardigan with no open front).
- not_applicable: the closure state cannot be seen (occluded, cropped out, or genuinely not visible).

## have_inner — whether another garment layer is visible underneath this outerwear

- true: a separate inner garment is visible under the outerwear (e.g. shirt/blouse/t-shirt/sweater under jacket/coat/cardigan).
- false: no separate inner layer is visible; only the outer garment is seen on the upper body.
- Do NOT output not_applicable here — is_outerwear is true, so have_inner must be true or false.

Output exactly one JSON object, no markdown:

{
  "layering_structure": {
    "outer_closure": "open",
    "have_inner": true
  }
}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process annotation JSONL rows.")
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-jsonl", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument("--base-url", default=os.environ.get("RIVO_BASE_URL", RIVO_DEFAULT_BASE_URL))
    parser.add_argument("--api-key", default=os.environ.get("RIVO_API_KEY", ""))
    parser.add_argument("--model", default=os.environ.get("RIVO_MODEL", RIVO_DEFAULT_MODEL))
    parser.add_argument(
        "--relabel-layering",
        action="store_true",
        help="Call Rivo to relabel outer_closure/have_inner when person outerwear is corrected.",
    )
    parser.add_argument(
        "--no-relabel-layering",
        action="store_true",
        help="Skip Rivo relabel even if --api-key is set.",
    )
    return parser.parse_args()


def resolve_script_path(path: Path) -> Path:
    if path.is_file():
        return path
    candidate = SCRIPT_DIR / path
    if candidate.is_file():
        return candidate
    return path


def resolve_row_image_path(row: dict[str, Any], field: str, *, dataset_root: Path) -> Path:
    value = row.get(field)
    if isinstance(value, dict):
        file_name = value.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            raise ValueError(f"{field}.file_name missing")
        path = Path(file_name)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} missing")

    path = Path(value)
    if path.is_absolute():
        return path

    project_candidate = PROJECT_ROOT / path
    if project_candidate.is_file():
        return project_candidate

    dataset_candidate = dataset_root / path
    if dataset_candidate.is_file():
        return dataset_candidate

    return project_candidate


def is_falsey_outerwear(value: Any) -> bool:
    return value is False or value == "false" or value is None


def normalize_neckline_to_round_neck(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {a.lower() for a in ROUND_NECK_ALIASES}:
        return "round_neck"
    return value


def normalize_tucking_to_half_tuck(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {a.lower() for a in HALF_TUCK_ALIASES}:
        return "half_tuck"
    return value


def normalize_edit_attribute_value(dimension: str, attribute: str, value: Any) -> Any:
    """Normalize single edit attribute values to match postprocess canonical enums."""
    if dimension == "wearing_style" and attribute == "tucking_style":
        return normalize_tucking_to_half_tuck(value)
    if dimension == "local_structure" and attribute == "neckline":
        return normalize_neckline_to_round_neck(value)
    return value


def sync_edit_changes_from_attributes(
    changes: dict[str, dict[str, dict[str, Any]]],
    *,
    source_attributes: dict[str, Any],
    target_attributes: dict[str, Any],
    active_region: str,
) -> None:
    """Align changes from/to with postprocessed source and target attribute blocks."""
    source_region = source_attributes.get(active_region)
    target_region = target_attributes.get(active_region)
    if not isinstance(source_region, dict) or not isinstance(target_region, dict):
        return
    for dimension, attrs in changes.items():
        if not attrs:
            continue
        source_block = source_region.get(dimension)
        target_block = target_region.get(dimension)
        for attribute, change in attrs.items():
            if isinstance(source_block, dict) and attribute in source_block:
                change["from"] = source_block[attribute]
            if isinstance(target_block, dict) and attribute in target_block:
                change["to"] = target_block[attribute]


def process_wearing_style(region: dict[str, Any]) -> None:
    wearing_style = region.get("wearing_style")
    if not isinstance(wearing_style, dict):
        return
    tucking = wearing_style.get("tucking_style")
    normalized = normalize_tucking_to_half_tuck(tucking)
    if normalized != tucking:
        wearing_style["tucking_style"] = normalized


def process_local_structure(region: dict[str, Any], *, region_key: str) -> None:
    category = region.get("category")
    if not isinstance(category, str):
        category = ""
    category = category.strip().lower()

    local_structure = region.setdefault("local_structure", {})

    neckline = local_structure.get("neckline")
    normalized_neckline = normalize_neckline_to_round_neck(neckline)
    if normalized_neckline != neckline:
        local_structure["neckline"] = normalized_neckline

    # 衬衫：按 category 强制 neckline=not_applicable
    if category in SHIRT_NECKLINE_NA_CATEGORIES:
        local_structure["neckline"] = "not_applicable"

    # 外套：layering 已处理完后，is_outerwear 不为 false 则 neckline=not_applicable
    if region_key in LAYERING_REGION_KEYS:
        layering = region.get("layering_structure")
        if isinstance(layering, dict) and not is_falsey_outerwear(layering.get("is_outerwear")):
            local_structure["neckline"] = "not_applicable"

    if category == "t-shirt":
        local_structure["collar_type"] = "not_applicable"
    elif category == "shirt":
        local_structure["collar_type"] = "shirt_collar"
    elif category == "coat":
        local_structure["collar_type"] = "lapel_collar"

    if local_structure.get("sleeve_length") == "sleeveless":
        local_structure["sleeve_type"] = "not_applicable"
        wearing_style = region.setdefault("wearing_style", {})
        wearing_style["sleeve_state"] = "not_applicable"


def process_layering_structure(
    region: dict[str, Any],
    *,
    region_key: str,
    is_person_upper: bool,
    layering_relabel_flag: dict[str, bool],
) -> None:
    if region_key not in LAYERING_REGION_KEYS:
        return

    layering = region.setdefault("layering_structure", {})
    category = region.get("category")
    if not isinstance(category, str):
        category = ""
    category = category.strip().lower()

    if category in OUTERWEAR_CATEGORIES and is_falsey_outerwear(layering.get("is_outerwear")):
        layering["is_outerwear"] = True
        if is_person_upper:
            layering_relabel_flag["needed"] = True

    if is_falsey_outerwear(layering.get("is_outerwear")):
        layering["outer_closure"] = "not_applicable"
        layering["have_inner"] = "not_applicable"
    elif layering.get("outer_closure") == "close":
        layering["outer_closure"] = "closed"


def process_region(
    region: dict[str, Any],
    *,
    region_key: str,
    is_person_upper: bool,
    layering_relabel_flag: dict[str, bool],
) -> None:
    if not isinstance(region, dict):
        return
    if region.get("is_present") is False:
        return
    if "category" not in region and "local_structure" not in region:
        return

    process_wearing_style(region)
    process_layering_structure(
        region,
        region_key=region_key,
        is_person_upper=is_person_upper,
        layering_relabel_flag=layering_relabel_flag,
    )
    process_local_structure(region, region_key=region_key)


def remove_visibility_and_occlusion(person_annotation: dict[str, Any]) -> None:
    person_annotation.pop("visibility", None)
    person_annotation.pop("occlusion_level", None)


def process_person_annotation(
    person_annotation: dict[str, Any],
    layering_relabel_flag: dict[str, bool],
) -> None:
    remove_visibility_and_occlusion(person_annotation)
    for region_key in REGION_KEYS:
        region = person_annotation.get(region_key)
        if not isinstance(region, dict):
            continue
        process_region(
            region,
            region_key=region_key,
            is_person_upper=region_key == "upper_body_garment",
            layering_relabel_flag=layering_relabel_flag,
        )


def process_garment_annotation(garment_annotation: dict[str, Any]) -> None:
    noop_flag: dict[str, bool] = {"needed": False}
    for region_key in REGION_KEYS:
        region = garment_annotation.get(region_key)
        if not isinstance(region, dict):
            continue
        process_region(
            region,
            region_key=region_key,
            is_person_upper=False,
            layering_relabel_flag=noop_flag,
        )


def process_attributes_block(block: dict[str, Any]) -> None:
    noop_flag: dict[str, bool] = {"needed": False}
    remove_visibility_and_occlusion(block)
    for region_key in REGION_KEYS:
        region = block.get(region_key)
        if not isinstance(region, dict):
            continue
        process_region(
            region,
            region_key=region_key,
            is_person_upper=region_key == "upper_body_garment",
            layering_relabel_flag=noop_flag,
        )


def sync_garment_layering_from_person(row: dict[str, Any]) -> None:
    """After person-side outerwear relabel, mirror closure fields to garment."""
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        return
    person_image = annotation.get("person_image")
    garment_image = annotation.get("garment_image")
    if not isinstance(person_image, dict) or not isinstance(garment_image, dict):
        return
    person_annotation = person_image.get("person_annotation")
    garment_annotation = garment_image.get("garment_annotation")
    if not isinstance(person_annotation, dict) or not isinstance(garment_annotation, dict):
        return
    if garment_annotation.get("where_to_dress") != "upper_body":
        return

    region_key = "upper_body_garment"
    person_region = person_annotation.get(region_key)
    garment_region = garment_annotation.get(region_key)
    if not isinstance(person_region, dict) or not isinstance(garment_region, dict):
        return
    if person_region.get("is_present") is False:
        return

    person_layer = person_region.get("layering_structure")
    if not isinstance(person_layer, dict):
        return

    garment_layer = garment_region.setdefault("layering_structure", {})
    garment_layer["outer_closure"] = person_layer.get("outer_closure")
    garment_layer["is_outerwear"] = person_layer.get("is_outerwear")
    process_local_structure(garment_region, region_key=region_key)


def validate_layering_relabel(payload: dict[str, Any]) -> dict[str, Any]:
    layering = payload.get("layering_structure")
    if not isinstance(layering, dict):
        raise ValueError("layering_structure missing")
    outer_closure = layering.get("outer_closure")
    if outer_closure not in ("open", "closed", "not_applicable"):
        raise ValueError(f"invalid outer_closure: {outer_closure!r}")
    have_inner = layering.get("have_inner")
    if have_inner not in (True, False, "true", "false"):
        raise ValueError(f"invalid have_inner: {have_inner!r}")
    if isinstance(have_inner, str):
        have_inner = have_inner.lower() == "true"
    return {"outer_closure": outer_closure, "have_inner": have_inner}


def relabel_person_upper_layering(
    row: dict[str, Any],
    *,
    dataset_root: Path,
    base_url: str,
    api_key: str,
    model: str,
) -> None:
    annotation = row["annotation"]
    person_annotation = annotation["person_image"]["person_annotation"]
    region = person_annotation["upper_body_garment"]
    category = region.get("category", "outerwear")

    person_path = resolve_row_image_path(row, "person_image", dataset_root=dataset_root)
    if not person_path.is_file():
        raise FileNotFoundError(f"person image not found: {person_path}")

    user_prompt = (
        f"The upper_body_garment category is {category}. "
        "Return layering_structure.outer_closure and layering_structure.have_inner only."
    )
    messages = [
        {"role": "system", "content": LAYERING_RELABEL_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1: model/person image"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(person_path)},
                },
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    raw_text = call_openai_chat_completions(
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=512,
        timeout=120.0,
        use_ipv4=False,
    )
    parsed = extract_json_object(raw_text)
    validated = validate_layering_relabel(parsed)
    layering = region.setdefault("layering_structure", {})
    layering["outer_closure"] = validated["outer_closure"]
    layering["have_inner"] = validated["have_inner"]
    layering["is_outerwear"] = True
    process_local_structure(region, region_key="upper_body_garment")
    sync_garment_layering_from_person(row)


def postprocess_row(row: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(row)
    layering_relabel_flag: dict[str, bool] = {"needed": False}

    annotation = merged.get("annotation")
    if isinstance(annotation, dict):
        person_image = annotation.get("person_image")
        if isinstance(person_image, dict):
            person_annotation = person_image.get("person_annotation")
            if isinstance(person_annotation, dict):
                process_person_annotation(person_annotation, layering_relabel_flag)

        garment_image = annotation.get("garment_image")
        if isinstance(garment_image, dict):
            garment_annotation = garment_image.get("garment_annotation")
            if isinstance(garment_annotation, dict):
                process_garment_annotation(garment_annotation)

    for key in ("source_attributes", "target_attributes"):
        block = merged.get(key)
        if isinstance(block, dict):
            process_attributes_block(block)

    meta = merged.setdefault("postprocess_meta", {})
    meta["postprocessed_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    meta["postprocess_script"] = "postprocess_annotations.py"
    if layering_relabel_flag["needed"]:
        meta["layering_relabel_needed"] = True
    return merged


def collect_stats(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    stats: dict[str, Counter[str]] = {
        "neckline": Counter(),
        "tucking": Counter(),
        "collar_type": Counter(),
        "outer_closure": Counter(),
        "is_outerwear": Counter(),
        "have_inner": Counter(),
        "garment_g1": Counter(),
        "visibility_present": Counter(),
    }

    def scan(container: dict[str, Any] | None) -> None:
        if not isinstance(container, dict):
            return
        if "visibility" in container:
            stats["visibility_present"]["yes"] += 1
        if "occlusion_level" in container:
            stats["visibility_present"]["occlusion_yes"] += 1
        for region_key in REGION_KEYS:
            region = container.get(region_key)
            if not isinstance(region, dict) or region.get("is_present") is False:
                continue
            ls = region.get("local_structure") or {}
            stats["neckline"][ls.get("neckline")] += 1
            stats["collar_type"][(region.get("category"), ls.get("collar_type"))] += 1
            tuck = (region.get("wearing_style") or {}).get("tucking_style")
            stats["tucking"][tuck] += 1
            layer = region.get("layering_structure") or {}
            if layer:
                stats["outer_closure"][layer.get("outer_closure")] += 1
                stats["is_outerwear"][layer.get("is_outerwear")] += 1
                stats["have_inner"][layer.get("have_inner")] += 1

    def scan_garment_g1(garment_annotation: dict[str, Any] | None) -> None:
        if not isinstance(garment_annotation, dict):
            return
        where = garment_annotation.get("where_to_dress")
        region_key = {
            "upper_body": "upper_body_garment",
            "lower_body": "lower_body_garment",
            "whole_body": "whole_body_garment",
        }.get(str(where), "")
        region = garment_annotation.get(region_key)
        if not isinstance(region, dict):
            return
        layer = region.get("layering_structure") or {}
        is_g1 = layer.get("is_outerwear") is True
        stats["garment_g1"][(where, region.get("category"), is_g1)] += 1

    for row in rows:
        ann = row.get("annotation", {})
        pa = ann.get("person_image", {}).get("person_annotation")
        ga = ann.get("garment_image", {}).get("garment_annotation")
        scan(pa)
        scan(ga)
        scan_garment_g1(ga)
        for key in ("source_attributes", "target_attributes"):
            scan(row.get(key))

    return stats


def print_stats(title: str, stats: dict[str, Counter[str]]) -> None:
    print(title, flush=True)
    print(f"  visibility/occlusion rows: {dict(stats['visibility_present'])}", flush=True)
    print(f"  necklines: {stats['neckline'].most_common(8)}", flush=True)
    print(f"  tucking: {stats['tucking'].most_common()}", flush=True)
    print(f"  outer_closure: {stats['outer_closure'].most_common()}", flush=True)
    print(f"  is_outerwear: {stats['is_outerwear'].most_common()}", flush=True)
    print(f"  have_inner: {stats['have_inner'].most_common()}", flush=True)
    shirt_collars = [(k, v) for k, v in stats["collar_type"].items() if k[0] == "shirt"]
    print(f"  shirt collar_type: {shirt_collars}", flush=True)
    g1_true = sum(v for (_, _, is_g1), v in stats["garment_g1"].items() if is_g1)
    print(f"  garment G1 rows: {g1_true}", flush=True)


def main() -> int:
    args = parse_args()
    args.input_jsonl = resolve_script_path(args.input_jsonl)

    rows = read_jsonl(args.input_jsonl)
    before = collect_stats(rows)
    processed = [postprocess_row(row) for row in rows]
    after = collect_stats(processed)

    print_stats("before", before)
    print_stats("after", after)

    relabel_count = 0
    relabel_failures: list[dict[str, Any]] = []
    should_relabel = (
        not args.dry_run
        and not args.no_relabel_layering
        and (args.relabel_layering or args.api_key)
    )
    if should_relabel and not args.api_key:
        print("relabel_layering skipped: RIVO_API_KEY / --api-key required", flush=True)
        should_relabel = False

    if should_relabel:
        for row in processed:
            meta = row.get("postprocess_meta", {})
            if not meta.get("layering_relabel_needed"):
                continue
            source_sample_id = str(row.get("source_sample_id", "unknown"))
            try:
                relabel_person_upper_layering(
                    row,
                    dataset_root=args.dataset_root,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    model=args.model,
                )
                meta["layering_relabeled_at"] = datetime.now(timezone.utc).isoformat().replace(
                    "+00:00", "Z"
                )
                meta.pop("layering_relabel_needed", None)
                relabel_count += 1
                print(f"[relabel-layering] ok source_sample_id={source_sample_id}", flush=True)
            except Exception as exc:  # noqa: BLE001
                relabel_failures.append(
                    {"source_sample_id": source_sample_id, "error": str(exc)}
                )
                print(
                    f"[relabel-layering] fail source_sample_id={source_sample_id} error={exc}",
                    flush=True,
                )

    if args.dry_run:
        pending = sum(
            1 for row in processed
            if row.get("postprocess_meta", {}).get("layering_relabel_needed")
        )
        print(f"dry_run=true no files written layering_relabel_pending={pending}", flush=True)
        return 0

    output_path = args.output_jsonl
    if output_path is None:
        output_path = args.input_jsonl.with_name(args.input_jsonl.stem + "_postprocessed.jsonl")
    elif not output_path.is_absolute():
        output_path = SCRIPT_DIR / output_path

    write_jsonl(output_path, processed)
    print(f"input={args.input_jsonl}", flush=True)
    print(f"output={output_path}", flush=True)
    print(f"rows={len(processed)}", flush=True)
    print(f"layering_relabeled={relabel_count} layering_relabel_failed={len(relabel_failures)}", flush=True)
    return 1 if relabel_failures else 0


if __name__ == "__main__":
    sys.exit(main())
