"""Scene-aware semantic constraints for attribute edits."""

from __future__ import annotations

from typing import Any

from edit_common import INVALID_VALUES, is_valid_value
from scene_classify import NO_TUCKING_STYLE_SCENES, person_wearing_type

OUTERWEAR_SCENES = frozenset({"L1", "L2", "L3", "E", "I1"})


def _upper_state(person_annotation: dict[str, Any]) -> str | None:
    _, upper_state = person_wearing_type(person_annotation)
    return upper_state


def has_inner_layer_for_outerwear_edit(
    *,
    person_annotation: dict[str, Any],
    scene_id: str | None,
) -> bool:
    """True when a false->true is_outerwear edit can truthfully say 'over inner'."""
    if scene_id == "L1":
        upper = person_annotation.get("upper_body_garment")
        return isinstance(upper, dict) and upper.get("is_present") is True
    if scene_id == "E":
        whole = person_annotation.get("whole_body_garment")
        return isinstance(whole, dict) and whole.get("is_present") is True
    if scene_id == "L2":
        return _upper_state(person_annotation) == "U2"
    if scene_id == "I1":
        upper = person_annotation.get("upper_body_garment")
        if not isinstance(upper, dict) or not upper.get("is_present"):
            return False
        layering = upper.get("layering_structure") or {}
        return layering.get("is_outerwear") is True
    return False


def is_layering_change_allowed(
    *,
    scene_id: str | None,
    person_annotation: dict[str, Any],
    attribute: str,
    from_value: Any,
    to_value: Any,
) -> bool:
    if attribute == "is_outerwear":
        if from_value is False and to_value is True:
            return has_inner_layer_for_outerwear_edit(
                person_annotation=person_annotation,
                scene_id=scene_id,
            )
        if from_value is True and to_value is False:
            if scene_id in OUTERWEAR_SCENES:
                return False
            return True

    if attribute == "outer_closure":
        if scene_id == "L3":
            return False
        if scene_id in {"I1", "I2"}:
            return False
        if to_value not in {"open", "closed"}:
            return False
        return True

    return True


def filter_editable_for_pair_mode(
    editable: list[tuple[str, str, Any]],
    *,
    pair_mode: str | None,
) -> list[tuple[str, str, Any]]:
    """Paired person/garment are the same sample — no inner/outer layering reinterpretation."""
    if pair_mode != "paired":
        return editable
    return [
        (dimension, attribute, from_value)
        for dimension, attribute, from_value in editable
        if dimension != "layering_structure"
    ]


def filter_editable_for_scene(
    editable: list[tuple[str, str, Any]],
    *,
    scene_id: str | None,
    person_annotation: dict[str, Any],
    enums: dict[str, Any],
) -> list[tuple[str, str, Any]]:
    filtered: list[tuple[str, str, Any]] = []
    for dimension, attribute, from_value in editable:
        # E/C/D scenes: tucking_style is not editable
        if attribute == "tucking_style" and scene_id in NO_TUCKING_STYLE_SCENES:
            continue
        allowed = enums.get(dimension, {}).get(attribute, [])
        has_valid_target = False
        for to_value in allowed:
            if to_value == from_value:
                continue
            if to_value in INVALID_VALUES:
                continue
            if str(to_value).lower() == "not_applicable":
                continue
            if is_layering_change_allowed(
                scene_id=scene_id,
                person_annotation=person_annotation,
                attribute=attribute,
                from_value=from_value,
                to_value=to_value,
            ):
                has_valid_target = True
                break
        if has_valid_target:
            filtered.append((dimension, attribute, from_value))
    return filtered
