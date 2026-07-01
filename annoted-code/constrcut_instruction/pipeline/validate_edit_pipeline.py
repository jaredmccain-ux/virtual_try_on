#!/usr/bin/env python3
"""Validate edit assignment quotas and instruction/target consistency."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PIPELINE_DIR.parent
DATA_DIR = ROOT_DIR / "data"
sys.path.insert(0, str(ROOT_DIR.parent))

from edit_common import REGION_FIELDS, is_valid_value  # noqa: E402
from postprocess_annotations import process_attributes_block  # noqa: E402

REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def check_quotas(rows: list[dict], catalog_path: Path, label: str) -> list[str]:
    catalog = json.loads(catalog_path.read_text())
    targets = {
        tid: int(catalog["types"][tid].get("target_count", catalog.get("default_target_count", 10)))
        for tid in catalog["types"]
    }
    counts = Counter(row["edit_task"]["edit_type_id"] for row in rows if row.get("edit_task"))
    issues = []
    for tid, target in sorted(targets.items()):
        cnt = counts.get(tid, 0)
        if cnt != target:
            issues.append(f"{label} {tid}: {cnt}/{target}")
    return issues


def schema_issues_person_lower(region: dict) -> list[str]:
    if not region.get("is_present"):
        return []
    missing = []
    for block in ("wearing_style", "fit_silhouette", "local_structure"):
        if not isinstance(region.get(block), dict):
            missing.append(f"missing {block}")
    app = region.get("appearance") or {}
    for k in ("color", "pattern", "material_texture", "embellishment"):
        if k not in app:
            missing.append(f"missing appearance.{k}")
    if region.get("category") is None:
        missing.append("category=null")
    return missing


def verify_target_matches_edits(row: dict) -> list[str]:
    issues = []
    edit_task = row.get("edit_task") or {}
    changes = edit_task.get("changes_by_dimension") or {}
    target = row.get("target_attributes") or {}
    active = row.get("active_region") or edit_task.get("region")
    region = target.get(active) if active else None
    if not isinstance(region, dict):
        return ["missing target active region"]

    for dimension, attrs in changes.items():
        if not attrs:
            continue
        block = region.get(dimension)
        if not isinstance(block, dict):
            issues.append(f"target missing block {dimension}")
            continue
        for attr, change in attrs.items():
            expected = change.get("to")
            actual = block.get(attr)
            if actual != expected:
                issues.append(f"{dimension}.{attr}: target={actual!r} edit_to={expected!r}")

    # Re-apply postprocess should be stable
    copy = json.loads(json.dumps(target))
    process_attributes_block(copy)
    if copy != target:
        issues.append("target not postprocess-stable")
    return issues


def extract_edit_phrases_en(instruction_en: str) -> list[tuple[str, str, str]]:
    """Parse 'from X to Y' in EDIT sections (rough)."""
    results = []
    for segment in re.findall(r"\[EDIT · [^\]]+\][^\.]+", instruction_en):
        for m in re.finditer(r"from ([^;]+?) to ([^;\.]+)", segment, re.I):
            results.append((segment.split("]")[0], m.group(1).strip(), m.group(2).strip()))
    return results


def main() -> int:
    catalog = PIPELINE_DIR / "edit_type_catalog.json"
    paired = load_jsonl(DATA_DIR / "annotations_paired_with_edit.jsonl")
    unpair = load_jsonl(DATA_DIR / "annotations_unpair_with_edit.jsonl")

    print("=== Assignment failures ===")
    for name in ("paired_edit_failures.jsonl", "unpair_edit_failures.jsonl"):
        p = DATA_DIR / name
        n = len(load_jsonl(p)) if p.is_file() and p.stat().st_size else 0
        print(f"  {name}: {n}")

    print("\n=== Quota mismatches (count != target_count) ===")
    for label, rows in [("paired", paired), ("unpair", unpair)]:
        mismatches = check_quotas(rows, catalog, label)
        if mismatches:
            print(f"  {label}: {len(mismatches)} types off target")
            for m in mismatches[:8]:
                print(f"    {m}")
            if len(mismatches) > 8:
                print(f"    ... +{len(mismatches) - 8} more")
        else:
            print(f"  {label}: all quotas exact")

    print("\n=== Person lower incomplete in source_attributes ===")
    for label, rows in [("paired", paired), ("unpair", unpair)]:
        bad = []
        for row in rows:
            lower = (row.get("source_attributes") or {}).get("lower_body_garment", {})
            iss = schema_issues_person_lower(lower)
            if iss:
                bad.append((row.get("sample_id"), iss))
        print(f"  {label}: {len(bad)} incomplete")

    print("\n=== target vs edit_task consistency ===")
    for label, rows in [("paired", paired), ("unpair", unpair)]:
        bad = []
        for row in rows:
            iss = verify_target_matches_edits(row)
            if iss:
                bad.append((row.get("sample_id"), iss))
        print(f"  {label}: {len(bad)} mismatches")
        if bad[:3]:
            for sid, iss in bad[:3]:
                print(f"    {sid}: {iss[:2]}")

    print("\n=== Spot check instructions (3 paired) ===")
    for row in paired[:3]:
        sid = row.get("sample_id")
        edit = row["edit_task"]["edit_type_id"]
        en = row["instruction"]["instruction_en"][:200]
        print(f"  {sid} [{edit}]: {en}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
