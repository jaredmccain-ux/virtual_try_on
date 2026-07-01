"""Shared helpers for edit assignment and instruction rendering."""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

DIMENSIONS = (
    "wearing_style",
    "fit_silhouette",
    "local_structure",
    "appearance",
    "layering_structure",
)

WHERE_TO_REGION = {
    "upper_body": "upper_body_garment",
    "lower_body": "lower_body_garment",
    "whole_body": "whole_body_garment",
}

REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")

REGION_FIELDS: dict[str, dict[str, list[str]]] = {
    "upper_body_garment": {
        "wearing_style": ["tucking_style", "sleeve_state"],
        "fit_silhouette": ["fit"],
        "local_structure": ["neckline", "sleeve_length", "sleeve_type", "collar_type"],
        "appearance": ["color", "pattern", "material_texture", "embellishment"],
        "layering_structure": ["outer_closure", "is_outerwear"],
    },
    "lower_body_garment": {
        "wearing_style": ["pants_cuff_state"],
        "fit_silhouette": ["fit"],
        "local_structure": ["hem_length", "hem_shape"],
        "appearance": ["color", "pattern", "material_texture", "embellishment"],
    },
    "whole_body_garment": {
        "wearing_style": ["tucking_style", "sleeve_state", "pants_cuff_state"],
        "fit_silhouette": ["fit"],
        "local_structure": [
            "neckline",
            "sleeve_length",
            "sleeve_type",
            "collar_type",
            "hem_length",
            "hem_shape",
        ],
        "appearance": ["color", "pattern", "material_texture", "embellishment"],
        "layering_structure": ["outer_closure", "is_outerwear"],
    },
}

INVALID_VALUES = {None, "not_applicable", "not_visible"}

PERSON_PRESERVE_KEYS = [
    "identity",
    "face",
    "pose",
    "body_shape",
    "background",
    "lighting",
]

ATTRIBUTE_LABEL_EN: dict[str, str] = {
    "tucking_style": "tucking style",
    "sleeve_state": "sleeve state",
    "pants_cuff_state": "pants cuff state",
    "fit": "fit",
    "neckline": "neckline",
    "sleeve_length": "sleeve length",
    "sleeve_type": "sleeve type",
    "collar_type": "collar type",
    "hem_length": "hem length",
    "hem_shape": "hem shape",
    "color": "color",
    "pattern": "pattern",
    "material_texture": "material texture",
    "embellishment": "embellishment",
    "outer_closure": "outerwear closure",
    "is_outerwear": "outerwear usage",
}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
            rows.append(row)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sample_key(row: dict[str, Any]) -> str:
    return str(
        row.get("sample_id")
        or row.get("source_sample_id")
        or row.get("person_anchor_id")
        or "unknown"
    )


def get_person_annotation(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("source_attributes"), dict):
        return copy.deepcopy(row["source_attributes"])
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"{sample_key(row)}: missing annotation")
    person_image = annotation.get("person_image")
    if not isinstance(person_image, dict):
        raise ValueError(f"{sample_key(row)}: missing annotation.person_image")
    person_annotation = person_image.get("person_annotation")
    if not isinstance(person_annotation, dict):
        raise ValueError(f"{sample_key(row)}: missing person_annotation")
    return copy.deepcopy(person_annotation)


def get_active_region(row: dict[str, Any]) -> str:
    active = row.get("active_region")
    if active in REGION_KEYS:
        return str(active)
    annotation = row["annotation"]
    garment_image = annotation["garment_image"]
    where = garment_image["garment_annotation"]["where_to_dress"]
    if where not in WHERE_TO_REGION:
        raise ValueError(f"{sample_key(row)}: invalid where_to_dress={where!r}")
    return WHERE_TO_REGION[where]


def serialize_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    return value


def is_valid_value(value: Any) -> bool:
    return value not in INVALID_VALUES


def layering_allows_attribute(region_state: dict[str, Any], attribute: str) -> bool:
    if attribute != "outer_closure":
        return True
    layering = region_state.get("layering_structure")
    if not isinstance(layering, dict):
        return False
    return layering.get("is_outerwear") is True


def region_supports_attribute(region: str, dimension: str, attribute: str) -> bool:
    fields = REGION_FIELDS.get(region, {})
    return attribute in fields.get(dimension, [])


def get_region_value(
    person_annotation: dict[str, Any], region: str, dimension: str, attribute: str
) -> Any:
    region_obj = person_annotation.get(region)
    if not isinstance(region_obj, dict) or not region_obj.get("is_present"):
        return None
    dimension_obj = region_obj.get(dimension)
    if not isinstance(dimension_obj, dict):
        return None
    return dimension_obj.get(attribute)


def get_garment_annotation(row: dict[str, Any]) -> dict[str, Any]:
    annotation = row.get("annotation")
    if not isinstance(annotation, dict):
        raise ValueError(f"{sample_key(row)}: missing annotation")
    garment_image = annotation.get("garment_image")
    if not isinstance(garment_image, dict):
        raise ValueError(f"{sample_key(row)}: missing annotation.garment_image")
    garment_annotation = garment_image.get("garment_annotation")
    if not isinstance(garment_annotation, dict):
        raise ValueError(f"{sample_key(row)}: missing garment_annotation")
    return copy.deepcopy(garment_annotation)


def infer_garment_wearing_style(
    region_state: dict[str, Any],
    active_region: str,
) -> dict[str, Any]:
    """Infer worn-state defaults for flat-lay garments missing wearing_style."""
    category = str(region_state.get("category") or "")
    local = region_state.get("local_structure") or {}
    sleeve_length = local.get("sleeve_length")
    inferred: dict[str, Any] = {}

    if active_region in {"upper_body_garment", "whole_body_garment"}:
        if sleeve_length == "sleeveless":
            inferred["sleeve_state"] = "not_applicable"
        elif sleeve_length not in INVALID_VALUES:
            inferred["sleeve_state"] = "sleeves_down"

    if active_region in {"lower_body_garment", "whole_body_garment"}:
        if category in {"pants", "shorts", "skirt"} or active_region == "lower_body_garment":
            inferred["pants_cuff_state"] = "normal_pants_cuff"
        elif active_region == "whole_body_garment":
            inferred["pants_cuff_state"] = "not_applicable"

    return inferred


def resolve_tucking_style(
    *,
    garment_region: dict[str, Any],
    person_region: dict[str, Any] | None,
    active_region: str,
) -> Any:
    """Garment first; if missing, fall back to the person's tucking_style."""
    garment_ws = garment_region.get("wearing_style") or {}
    if isinstance(garment_ws, dict) and is_valid_value(garment_ws.get("tucking_style")):
        return garment_ws["tucking_style"]

    if isinstance(person_region, dict):
        person_ws = person_region.get("wearing_style") or {}
        if isinstance(person_ws, dict) and is_valid_value(person_ws.get("tucking_style")):
            return person_ws["tucking_style"]

    category = str(garment_region.get("category") or "")
    if category in {"dress", "pants", "skirt", "shorts"}:
        return "not_applicable"
    return "untucked"


def resolve_embellishment(garment_region: dict[str, Any]) -> str:
    """Use flat-lay garment only; default to none when absent."""
    appearance = garment_region.get("appearance") or {}
    if isinstance(appearance, dict):
        value = appearance.get("embellishment")
        if is_valid_value(value):
            return str(value)
    return "none"


def build_active_region_edit_state(
    person_annotation: dict[str, Any],
    garment_annotation: dict[str, Any],
    active_region: str,
) -> dict[str, Any]:
    """Build post-try-on baseline for active_region from the given garment flat lay."""
    garment_region = garment_annotation.get(active_region)
    if not isinstance(garment_region, dict):
        raise ValueError(f"garment missing active region block: {active_region}")

    person_region = person_annotation.get(active_region)
    person_region = person_region if isinstance(person_region, dict) else None

    state: dict[str, Any] = {
        "is_present": True,
        "category": garment_region.get("category"),
    }
    for dimension, attributes in REGION_FIELDS.get(active_region, {}).items():
        if dimension == "wearing_style":
            continue
        source_block = garment_region.get(dimension)
        if not isinstance(source_block, dict):
            source_block = {}
        kept: dict[str, Any] = {}
        for attribute in attributes:
            if attribute == "embellishment":
                kept[attribute] = resolve_embellishment(garment_region)
                continue
            value = source_block.get(attribute)
            if is_valid_value(value):
                kept[attribute] = value
        if kept:
            state[dimension] = kept

    wearing_style: dict[str, Any] = {}
    ws_fields = REGION_FIELDS.get(active_region, {}).get("wearing_style", [])
    if "tucking_style" in ws_fields:
        wearing_style["tucking_style"] = resolve_tucking_style(
            garment_region=garment_region,
            person_region=person_region,
            active_region=active_region,
        )
    for attribute, value in infer_garment_wearing_style(state, active_region).items():
        if attribute in ws_fields and attribute != "tucking_style":
            if is_valid_value(value):
                wearing_style[attribute] = value
    if wearing_style:
        state["wearing_style"] = wearing_style

    return state


def attribute_applicable_on_region(region_state: dict[str, Any], dimension: str, attribute: str) -> bool:
    """Drop edits that contradict the garment structure (e.g. sleeve_state on sleeveless)."""
    if not layering_allows_attribute(region_state, attribute):
        return False
    local = region_state.get("local_structure") or {}
    sleeve_length = local.get("sleeve_length")
    if attribute == "sleeve_state":
        return sleeve_length not in INVALID_VALUES and sleeve_length != "sleeveless"
    if attribute == "collar_type":
        if local.get("collar_type") in INVALID_VALUES:
            return False
        category = str(region_state.get("category") or "").strip().lower()
        return category not in {"t-shirt", "shirt", "coat"}
    if attribute == "sleeve_type":
        return sleeve_length not in INVALID_VALUES and sleeve_length != "sleeveless"
    if attribute == "tucking_style":
        category = str(region_state.get("category") or "")
        return category not in {"dress", "pants", "skirt", "shorts"}
    return True


def list_editable_attributes(
    person_annotation: dict[str, Any],
    garment_annotation: dict[str, Any],
    active_region: str,
) -> list[tuple[str, str, Any]]:
    editable: list[tuple[str, str, Any]] = []
    edit_state = build_active_region_edit_state(
        person_annotation, garment_annotation, active_region
    )
    for dimension, attributes in REGION_FIELDS.get(active_region, {}).items():
        for attribute in attributes:
            if not region_supports_attribute(active_region, dimension, attribute):
                continue
            block = edit_state.get(dimension)
            if not isinstance(block, dict):
                continue
            value = block.get(attribute)
            if not is_valid_value(value):
                continue
            if not attribute_applicable_on_region(edit_state, dimension, attribute):
                continue
            editable.append((dimension, attribute, value))
    return editable


def pick_to_value(
    *,
    dimension: str,
    attribute: str,
    from_value: Any,
    enums: dict[str, Any],
    rng: random.Random,
) -> Any | None:
    from postprocess_annotations import normalize_edit_attribute_value

    from_value = normalize_edit_attribute_value(dimension, attribute, from_value)
    allowed = enums.get(dimension, {}).get(attribute, [])
    candidates: list[Any] = []
    for item in allowed:
        item_norm = normalize_edit_attribute_value(dimension, attribute, item)
        if item_norm == from_value:
            continue
        if item in INVALID_VALUES:
            continue
        if str(item).lower() in {"not_applicable", "none"}:
            continue
        candidates.append(item)
    if not candidates:
        return None
    return normalize_edit_attribute_value(dimension, attribute, rng.choice(candidates))


def empty_changes_by_dimension() -> dict[str, dict[str, dict[str, Any]]]:
    return {dimension: {} for dimension in DIMENSIONS}


def humanize(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("_", " ")


def attribute_label(attribute: str) -> str:
    return ATTRIBUTE_LABEL_EN.get(attribute, attribute.replace("_", " "))


def _layering_closure_value(layering: dict[str, Any]) -> Any:
    closure = layering.get("outer_closure")
    if closure in {"open", "closed"}:
        return closure
    return None


def render_layering_preserve_en(layering: dict[str, Any]) -> str | None:
    """Natural-language preserve phrase; omit default non-outerwear states."""
    if layering.get("is_outerwear") is not True:
        return None
    closure = _layering_closure_value(layering)
    if closure == "open":
        return "keep it worn as open outerwear"
    if closure == "closed":
        return "keep it worn as fully closed outerwear"
    return "keep it worn as outerwear"


def render_layering_preserve_zh(layering: dict[str, Any]) -> str | None:
    if layering.get("is_outerwear") is not True:
        return None
    closure = _layering_closure_value(layering)
    if closure == "open":
        return "保持作为敞开式外层穿着"
    if closure == "closed":
        return "保持作为扣合式外层穿着"
    return "保持作为外层穿着"


def render_layering_edit_en(
    attribute: str,
    from_value: Any,
    to_value: Any,
    *,
    scene_id: str | None = None,
) -> str:
    if attribute == "is_outerwear":
        if from_value is False and to_value is True:
            if scene_id == "L1":
                return "wear it layered over the existing upper-body garment"
            if scene_id == "E":
                return "wear it layered over the existing one-piece garment"
            if scene_id == "L2":
                return (
                    "wear it as the new outer layer while keeping the inner "
                    "upper-body garment unchanged"
                )
            return "wear it as an outer layer over the inner clothing"
        if from_value is True and to_value is False:
            return "wear it as a base layer instead of outerwear"
    if attribute == "outer_closure":
        if to_value == "open":
            return "wear the outerwear open"
        if to_value == "closed":
            return "wear the outerwear fully closed"
    return (
        f"change the {attribute_label(attribute)} from {humanize(from_value)} "
        f"to {humanize(to_value)}"
    )


def render_layering_edit_zh(
    attribute: str,
    from_value: Any,
    to_value: Any,
    *,
    scene_id: str | None = None,
) -> str:
    if attribute == "is_outerwear":
        if from_value is False and to_value is True:
            if scene_id == "L1":
                return "将其叠穿在现有上装之上"
            if scene_id == "E":
                return "将其叠穿在现有连体衣之上"
            if scene_id == "L2":
                return "将其作为新的外层穿着，并保持原有内搭上装不变"
            return "将其作为外层穿着，套在内搭之上"
        if from_value is True and to_value is False:
            return "将其作为内搭穿着，不要当作外层"
    if attribute == "outer_closure":
        if to_value == "open":
            return "将外层穿着方式改为敞开"
        if to_value == "closed":
            return "将外层穿着方式改为完全扣合"
    return (
        f"将{attribute_label(attribute)}从 {humanize(from_value)} "
        f"改为 {humanize(to_value)}"
    )
