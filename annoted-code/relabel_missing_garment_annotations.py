#!/usr/bin/env python3
"""Re-label garment_image for rows missing garment_annotation (VLM garment-only)."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from annotate_api import (  # noqa: E402
    PROJECT_ROOT,
    RIVO_DEFAULT_BASE_URL,
    RIVO_DEFAULT_MODEL,
    append_raw_record,
    call_openai_chat_completions,
    extract_json_object,
    format_raw_record,
    image_path_to_data_url,
    read_jsonl,
    write_jsonl,
)

DEFAULT_INPUT = SCRIPT_DIR / "annotations_api.jsonl"

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

UPPER_GARMENT_SYSTEM = """You are a fashion image analyst. You will see ONE image: a flat-lay upper-body garment.

Output exactly one JSON object (no markdown):

{
  "garment_image": {
    "garment_annotation": {
      "gender": "female",
      "where_to_dress": "upper_body",
      "upper_body_garment": {
        "category": "t-shirt",
        "local_structure": {
          "neckline": "round_neck",
          "sleeve_length": "short_sleeve",
          "sleeve_type": "regular",
          "collar_type": "not_applicable"
        },
        "appearance": {
          "color": "white",
          "pattern": "solid",
          "material_texture": "knit",
          "embellishment": "none"
        },
        "layering_structure": {
          "outer_closure": "not_applicable",
          "is_outerwear": false
        }
      }
    }
  }
}

Rules:
- category: shirt / blouse / t-shirt / sweater / tank_top / jacket / coat / cardigan / vest
- gender: male / female (menswear vs womenswear on the flat-lay image)
- where_to_dress must be "upper_body"
- Output only upper_body_garment under garment_annotation (no lower/whole blocks)
- embellishment required; use none when no obvious decoration
- If unclear, use not_visible for appearance/material fields
"""

LOWER_GARMENT_SYSTEM = """You are a fashion image analyst. You will see ONE image: a flat-lay lower-body garment.

Output exactly one JSON object (no markdown):

{
  "garment_image": {
    "garment_annotation": {
      "gender": "female",
      "where_to_dress": "lower_body",
      "lower_body_garment": {
        "category": "pants",
        "local_structure": {
          "hem_length": "full-length",
          "hem_shape": "straight_hem"
        },
        "appearance": {
          "color": "blue",
          "pattern": "solid",
          "material_texture": "denim",
          "embellishment": "none"
        }
      }
    }
  }
}

Rules:
- category: pants / skirt / shorts
- gender: male / female (menswear vs womenswear on the flat-lay image)
- where_to_dress must be "lower_body"
- Output only lower_body_garment under garment_annotation
- Do not output wearing_style, fit_silhouette, or layering_structure
- embellishment required; use none when no obvious decoration
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-label missing garment_image blocks and merge into annotations JSONL."
    )
    parser.add_argument("--input-jsonl", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--ids",
        default="",
        help="Comma-separated source_sample_ids (default: auto-detect missing garment_image).",
    )
    parser.add_argument("--api-key", default=os.environ.get("RIVO_API_KEY", ""))
    parser.add_argument("--base-url", default=os.environ.get("RIVO_BASE_URL", RIVO_DEFAULT_BASE_URL))
    parser.add_argument("--model", default=os.environ.get("RIVO_MODEL", RIVO_DEFAULT_MODEL))
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--raw-output-file",
        type=Path,
        default=SCRIPT_DIR / "model_rawsay" / "rivo__gpt-5.4__relabel_missing_garment.txt",
    )
    parser.add_argument(
        "--failure-log-jsonl",
        type=Path,
        default=SCRIPT_DIR / "relabel_missing_garment_failures.jsonl",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def infer_where_to_dress(source_sample_id: str) -> str:
    prefix = source_sample_id.split("_", 1)[0]
    if prefix == "upper":
        return "upper_body"
    if prefix == "lower":
        return "lower_body"
    if prefix == "whole":
        return "whole_body"
    raise ValueError(f"cannot infer where_to_dress from id: {source_sample_id}")


def has_garment_annotation(annotation: dict[str, Any]) -> bool:
    garment_image = annotation.get("garment_image")
    if not isinstance(garment_image, dict):
        return False
    garment_annotation = garment_image.get("garment_annotation")
    if not isinstance(garment_annotation, dict):
        return False
    where = garment_annotation.get("where_to_dress")
    if where not in ("upper_body", "lower_body", "whole_body"):
        return False
    region_key = {
        "upper_body": "upper_body_garment",
        "lower_body": "lower_body_garment",
        "whole_body": "whole_body_garment",
    }[where]
    region = garment_annotation.get(region_key)
    return isinstance(region, dict) and bool(region)


def normalize_person_image_block(annotation: dict[str, Any]) -> dict[str, Any]:
    """Flattened {role, wearing_type, person_annotation} -> {person_image: ...}."""
    if "person_image" in annotation:
        return annotation
    if "person_annotation" not in annotation:
        return annotation
    person_image = {
        "role": annotation.get("role"),
        "wearing_type": annotation.get("wearing_type"),
        "person_annotation": copy.deepcopy(annotation["person_annotation"]),
    }
    normalized = {"person_image": person_image}
    if "garment_image" in annotation:
        normalized["garment_image"] = copy.deepcopy(annotation["garment_image"])
    return normalized


def person_role_from_annotation(annotation: dict[str, Any]) -> str | None:
    person_image = annotation.get("person_image")
    if isinstance(person_image, dict) and person_image.get("role"):
        return str(person_image["role"])
    role = annotation.get("role")
    return str(role) if role else None


def resolve_garment_path(row: dict[str, Any]) -> Path:
    garment_image = row.get("garment_image")
    if not isinstance(garment_image, str) or not garment_image.strip():
        raise ValueError("row.garment_image path missing")
    path = Path(garment_image)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise FileNotFoundError(f"garment image not found: {path}")
    return path


def parse_ids_arg(ids_arg: str) -> list[str]:
    if not ids_arg.strip():
        return list(KNOWN_MISSING_IDS)
    return [part.strip() for part in ids_arg.split(",") if part.strip()]


def detect_missing_ids(rows: list[dict[str, Any]]) -> list[str]:
    missing: list[str] = []
    for row in rows:
        annotation = row.get("annotation")
        if not isinstance(annotation, dict):
            continue
        if not has_garment_annotation(annotation):
            missing.append(str(row["source_sample_id"]))
    return missing


def extract_garment_image_block(parsed: dict[str, Any], where: str) -> dict[str, Any]:
    if isinstance(parsed.get("garment_image"), dict):
        garment_image = copy.deepcopy(parsed["garment_image"])
        ga = garment_image.get("garment_annotation")
        if not isinstance(ga, dict):
            raise ValueError("garment_image.garment_annotation missing in VLM output")
        ga["where_to_dress"] = where
        return garment_image

    region_key = {
        "upper_body": "upper_body_garment",
        "lower_body": "lower_body_garment",
        "whole_body": "whole_body_garment",
    }[where]
    if isinstance(parsed.get(region_key), dict):
        return {
            "garment_annotation": {
                "where_to_dress": where,
                region_key: copy.deepcopy(parsed[region_key]),
            }
        }

    raise ValueError(f"VLM output missing garment_image or {region_key}")


def ensure_garment_gender(garment_image: dict[str, Any], fallback_role: str | None) -> None:
    ga = garment_image.setdefault("garment_annotation", {})
    if not ga.get("gender") and fallback_role:
        ga["gender"] = fallback_role


def relabel_one_row(
    row: dict[str, Any],
    *,
    api_key: str,
    base_url: str,
    model: str,
    max_tokens: int,
    timeout: float,
    raw_output_file: Path | None,
) -> tuple[dict[str, Any], str]:
    source_sample_id = str(row["source_sample_id"])
    where = infer_where_to_dress(source_sample_id)
    system_prompt = UPPER_GARMENT_SYSTEM if where == "upper_body" else LOWER_GARMENT_SYSTEM
    garment_path = resolve_garment_path(row)
    person_image_path = row.get("person_image", "")

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"source_sample_id: {source_sample_id}"},
                {"type": "image_url", "image_url": {"url": image_path_to_data_url(garment_path)}},
                {
                    "type": "text",
                    "text": "Annotate this flat-lay garment image only. Output JSON as specified.",
                },
            ],
        },
    ]

    raw_text = call_openai_chat_completions(
        messages=messages,
        base_url=base_url,
        api_key=api_key,
        model=model,
        max_tokens=max_tokens,
        timeout=timeout,
        use_ipv4=False,
    )
    parsed = extract_json_object(raw_text)
    garment_image = extract_garment_image_block(parsed, where)
    ensure_garment_gender(garment_image, person_role_from_annotation(row.get("annotation", {})))

    if raw_output_file is not None:
        record = format_raw_record(
            source_sample_id=source_sample_id,
            model_id=model,
            person_image=str(person_image_path),
            garment_image=str(row.get("garment_image", "")),
            raw_text=raw_text,
            parsed=parsed,
        )
        append_raw_record(raw_output_file, record)

    updated = copy.deepcopy(row)
    annotation = updated.setdefault("annotation", {})
    annotation = normalize_person_image_block(annotation)
    annotation["garment_image"] = garment_image
    updated["annotation"] = annotation
    updated["relabel_garment_meta"] = {
        "relabel_missing_garment_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "where_to_dress": where,
        "model": model,
    }
    return updated, raw_text


def main() -> int:
    args = parse_args()
    if not args.api_key and not args.dry_run:
        print("ERROR: RIVO_API_KEY or --api-key required", flush=True)
        return 1

    input_path = args.input_jsonl
    if not input_path.is_file():
        print(f"ERROR: input not found: {input_path}", flush=True)
        return 1

    rows = read_jsonl(input_path)
    by_id = {str(r["source_sample_id"]): i for i, r in enumerate(rows)}

    target_ids = parse_ids_arg(args.ids)
    if args.ids.strip():
        work_ids = target_ids
    else:
        work_ids = detect_missing_ids(rows)
        if not work_ids:
            work_ids = list(KNOWN_MISSING_IDS)

    if args.limit > 0:
        work_ids = work_ids[: args.limit]

    print(f"input={input_path} relabel_targets={len(work_ids)}", flush=True)
    for sid in work_ids:
        print(f"  - {sid}", flush=True)

    if args.dry_run:
        print("dry_run=true no API calls, no file written", flush=True)
        return 0

    failures: list[dict[str, Any]] = []
    ok_count = 0

    for sid in work_ids:
        if sid not in by_id:
            failures.append({"source_sample_id": sid, "error": "not found in input jsonl"})
            print(f"[skip] {sid}: not in input", flush=True)
            continue

        row = rows[by_id[sid]]
        annotation = row.get("annotation", {})
        if has_garment_annotation(annotation) and args.ids.strip():
            print(f"[skip] {sid}: already has garment_annotation", flush=True)
            continue

        try:
            print(f"[relabel] {sid} ...", flush=True)
            updated, _ = relabel_one_row(
                row,
                api_key=args.api_key,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                raw_output_file=args.raw_output_file,
            )
            rows[by_id[sid]] = updated
            ok_count += 1
            print(f"[ok] {sid}", flush=True)
            time.sleep(0.5)
        except Exception as exc:  # noqa: BLE001
            failures.append({"source_sample_id": sid, "error": str(exc)})
            print(f"[fail] {sid}: {exc}", flush=True)

    write_jsonl(input_path, rows)
    print(f"written: {input_path} ({ok_count} rows updated)", flush=True)

    if failures:
        write_jsonl(args.failure_log_jsonl, failures)
        print(f"failures: {args.failure_log_jsonl} ({len(failures)})", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
