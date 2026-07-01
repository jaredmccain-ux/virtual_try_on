"""Scene classification and scene-aware instruction fragments (A/B/C/D/E/L1/L2/L3/I1/I2)."""

from __future__ import annotations

from typing import Any

WHERE_TO_REGION = {
    "upper_body": "upper_body_garment",
    "lower_body": "lower_body_garment",
    "whole_body": "whole_body_garment",
}
REGION_KEYS = ("upper_body_garment", "lower_body_garment", "whole_body_garment")

VALID_SCENES = frozenset({"A", "B", "C", "D", "E", "L1", "L2", "L3", "I1", "I2", "F"})

# Replace upper with flat-lay garment: preserve only the new garment block.
GARMENT_REPLACE_UPPER_SCENES = frozenset({"A", "I2"})

# Replace outfit with flat-lay one-piece: preserve only the new whole-body block.
GARMENT_REPLACE_WHOLE_SCENES = frozenset({"D"})

# Scenes where tucking_style is not meaningful in instructions.
# C/D: one-piece outcomes; E: layering (outerwear over existing) — tucking irrelevant.
NO_TUCKING_STYLE_SCENES = frozenset({"C", "D", "E"})

DEFAULT_SCENE_QUOTAS: dict[str, int] = {
    "I1": 0,
    "I2": 5,
    "L2": 7,
    "L1": 8,
    "L3": 8,
    "E": 6,
    "C": 22,
    "D": 22,
    "B": 28,
    "A": 94,
}

UNPAIR_SCENE_ASSIGN_ORDER = (
    "I2",
    "L2",
    "I1",
    "L1",
    "L3",
    "E",
    "C",
    "D",
    "B",
    "A",
)

PAIRED_ALLOWED_SCENES = frozenset({"A", "B", "C", "L2", "I1"})

PAIRED_SCENE_PRIORITY = (
    "L2",
    "I1",
    "C",
    "B",
    "A",
)


def is_truthy(value: Any) -> bool:
    return value is True or value == "true"


def person_wearing_type(person_annotation: dict[str, Any]) -> tuple[str, str | None]:
    """Return (wear_kind, upper_state) where upper_state is U0/U1/U2 or None."""
    whole = person_annotation.get("whole_body_garment")
    if isinstance(whole, dict) and whole.get("is_present"):
        return "whole", None

    upper = person_annotation.get("upper_body_garment")
    if not isinstance(upper, dict) or not upper.get("is_present"):
        return "invalid", None

    layering = upper.get("layering_structure") or {}
    if layering.get("is_outerwear") is not True:
        return "two_pieces", "U0"

    have_inner = layering.get("have_inner")
    if is_truthy(have_inner):
        return "two_pieces", "U2"
    return "two_pieces", "U1"


def garment_target(person_annotation: dict[str, Any], garment_annotation: dict[str, Any]) -> tuple[str, str, str | None]:
    """Return (where_to_dress, garment_class G0/G1, active_region)."""
    where = garment_annotation.get("where_to_dress")
    if where not in WHERE_TO_REGION:
        raise ValueError(f"invalid where_to_dress: {where!r}")
    region_key = WHERE_TO_REGION[str(where)]
    region = garment_annotation.get(region_key)
    if not isinstance(region, dict):
        raise ValueError(f"missing garment region block: {region_key}")
    layering = region.get("layering_structure") or {}
    garment_class = "G1" if layering.get("is_outerwear") is True else "G0"
    category = region.get("category")
    return str(where), garment_class, str(category) if category else None


def active_region_from_garment(garment_annotation: dict[str, Any]) -> str:
    where = garment_annotation["where_to_dress"]
    return WHERE_TO_REGION[str(where)]


def gender_compatible(*, person_role: Any, garment_gender: Any) -> bool:
    if not person_role or not garment_gender:
        return True
    return str(person_role) == str(garment_gender)


def eligible_scenes(
    person_annotation: dict[str, Any],
    garment_annotation: dict[str, Any],
) -> set[str]:
    wear_kind, upper_state = person_wearing_type(person_annotation)
    where, garment_class, _ = garment_target(person_annotation, garment_annotation)
    lower_present = (
        isinstance(person_annotation.get("lower_body_garment"), dict)
        and person_annotation["lower_body_garment"].get("is_present") is True
    )

    scenes: set[str] = set()

    if wear_kind == "whole" and where == "lower_body":
        scenes.add("F")
        return scenes

    if wear_kind == "two_pieces" and upper_state == "U0" and where == "upper_body" and garment_class == "G0":
        scenes.add("A")
    if lower_present and where == "lower_body" and garment_class == "G0":
        scenes.add("B")
    if wear_kind == "whole" and where == "whole_body" and garment_class == "G0":
        scenes.add("C")
    if wear_kind == "two_pieces" and where == "whole_body" and garment_class == "G0":
        scenes.add("D")
    if wear_kind == "whole" and garment_class == "G1":
        scenes.add("E")
    if wear_kind == "two_pieces" and upper_state == "U0" and where == "upper_body" and garment_class == "G1":
        scenes.update({"L1", "L3"})
    if (
        wear_kind == "two_pieces"
        and upper_state in {"U1", "U2"}
        and where == "upper_body"
        and garment_class == "G1"
    ):
        scenes.add("L2")
    if wear_kind == "two_pieces" and upper_state == "U2" and where == "upper_body" and garment_class == "G0":
        scenes.add("I1")
    if wear_kind == "two_pieces" and upper_state == "U1" and where == "upper_body" and garment_class == "G0":
        scenes.add("I2")

    return scenes


def eligible_paired_scenes(
    person_annotation: dict[str, Any],
    garment_annotation: dict[str, Any],
) -> set[str]:
    """Paired person/garment come from the same sample — no outerwear-on-outerwear layering."""
    scenes = eligible_scenes(person_annotation, garment_annotation) & PAIRED_ALLOWED_SCENES
    where, _, _ = garment_target(person_annotation, garment_annotation)
    if where == "whole_body":
        # Edit the one-piece flat lay itself (never scene E/D layering/replacement).
        scenes.add("C")
    if not scenes:
        # Trust garment target when person/garment annotations disagree on structure.
        if where == "upper_body":
            scenes.add("A")
        elif where == "lower_body":
            scenes.add("B")
        elif where == "whole_body":
            scenes.add("C")
    return scenes


def classify_paired_scene(
    person_annotation: dict[str, Any],
    garment_annotation: dict[str, Any],
) -> str | None:
    scenes = eligible_paired_scenes(person_annotation, garment_annotation)
    scenes.discard("F")
    if not scenes:
        return None
    for scene_id in PAIRED_SCENE_PRIORITY:
        if scene_id in scenes:
            return scene_id
    return None


def closure_phrase_en(
    garment_annotation: dict[str, Any],
    *,
    scene_id: str,
) -> str:
    if scene_id == "L3":
        return "worn fully closed"
    active = active_region_from_garment(garment_annotation)
    region = garment_annotation.get(active) or {}
    layering = region.get("layering_structure") or {}
    closure = layering.get("outer_closure")
    if closure == "open":
        return "worn open"
    if closure == "closed":
        return "worn fully closed"
    return "worn fully closed"


def render_scene_base_en(
    scene_id: str,
    garment_annotation: dict[str, Any],
) -> str:
    closure = closure_phrase_en(garment_annotation, scene_id=scene_id)
    templates = {
        "A": "Replace the person's upper-body garment with the given garment image.",
        "B": "Replace the person's lower-body garment with the given garment image.",
        "C": "Replace the person's one-piece garment with the given garment image.",
        "D": "Replace the person's outfit with the given one-piece garment image.",
        "E": f"Layer the given outerwear over the person's one-piece garment, {closure}.",
        "L1": f"Layer the given outerwear over the person's upper-body garment, {closure}.",
        "L2": f"Replace the person's outerwear with the given outerwear, {closure}.",
        "L3": "Replace the person's upper-body garment with the given outerwear, worn fully closed.",
        "I1": (
            "Replace the inner upper-body garment under the outerwear with the given "
            "garment image, keeping the outerwear unchanged."
        ),
        "I2": (
            "Remove the person's outerwear and replace the upper-body garment with the "
            "given garment image."
        ),
    }
    return templates[scene_id]


def render_scene_base_zh(
    scene_id: str,
    garment_annotation: dict[str, Any],
) -> str:
    closure = closure_phrase_en(garment_annotation, scene_id=scene_id)
    templates = {
        "A": "将人物的上装替换为给定平铺图衣物。",
        "B": "将人物的下装替换为给定平铺图衣物。",
        "C": "将人物的连体衣替换为给定平铺图衣物。",
        "D": "将人物的整体穿着替换为给定连体平铺图衣物。",
        "E": f"将给定外套叠穿在人物的连体衣之上，{closure}。",
        "L1": f"将给定外套叠穿在人物的上装之上，{closure}。",
        "L2": f"将人物的外套替换为给定外套，{closure}。",
        "L3": "将人物的上装替换为给定外套，并完全扣合穿着。",
        "I1": "在保留外套不变的前提下，将外层之下的内搭上装替换为给定平铺图衣物。",
        "I2": "去掉人物的外套，并将上装替换为给定平铺图衣物。",
    }
    return templates[scene_id]


def edit_subject_prefix_en(scene_id: str) -> str:
    if scene_id == "I1":
        return "On the newly worn inner upper-body garment, "
    if scene_id in {"L1", "E"}:
        return "On the newly layered outerwear, "
    if scene_id in {"L2", "L3"}:
        return "On the newly worn outerwear, "
    return "On the newly worn upper-body garment, "


def edit_subject_prefix_zh(scene_id: str) -> str:
    if scene_id == "I1":
        return "在新换上的内搭上装上，"
    if scene_id in {"L1", "E"}:
        return "在新叠穿的外套上，"
    if scene_id in {"L2", "L3"}:
        return "在新换上的外套上，"
    return "在新换上的上装上，"


def _drop_preserve_attribute(
    preserved: dict[str, Any],
    *,
    dimension: str,
    attribute: str,
) -> None:
    for region_preserved in preserved.values():
        if not isinstance(region_preserved, dict):
            continue
        block = region_preserved.get(dimension)
        if not isinstance(block, dict):
            continue
        block.pop(attribute, None)
        if not block:
            region_preserved.pop(dimension, None)


def adjust_preserved_for_scene(
    preserved: dict[str, Any],
    *,
    scene_id: str,
    source_attributes: dict[str, Any],
    active_region: str | None = None,
    pair_mode: str | None = None,
) -> dict[str, Any]:
    """Apply scene-specific preserve filtering on top of default preserved blocks."""
    adjusted = {key: value for key, value in preserved.items()}

    if scene_id == "E" and active_region == "upper_body_garment":
        # One-piece under the new outerwear: do not preserve inner whole-body attrs
        # (e.g. sleeveless) that contradict the layered outerwear scenario.
        adjusted.pop("whole_body_garment", None)

    if (
        scene_id in GARMENT_REPLACE_UPPER_SCENES
        and active_region == "upper_body_garment"
    ):
        # New upper comes from the flat-lay garment; whole_body is irrelevant in
        # two-piece scenarios, but lower_body should still be preserved.
        adjusted.pop("whole_body_garment", None)

    if (
        scene_id in GARMENT_REPLACE_WHOLE_SCENES
        and active_region == "whole_body_garment"
    ):
        # Two-piece -> one-piece: only describe the new dress/jumpsuit scenario.
        for region_key in REGION_KEYS:
            if region_key != active_region:
                adjusted.pop(region_key, None)

    if scene_id in NO_TUCKING_STYLE_SCENES:
        _drop_preserve_attribute(
            adjusted, dimension="wearing_style", attribute="tucking_style"
        )

    if pair_mode == "paired" and scene_id in {"A", "B", "C"}:
        for region_preserved in adjusted.values():
            if isinstance(region_preserved, dict):
                region_preserved.pop("layering_structure", None)

    upper = source_attributes.get("upper_body_garment")
    upper_is_outer = (
        isinstance(upper, dict)
        and upper.get("is_present")
        and (upper.get("layering_structure") or {}).get("is_outerwear") is True
    )

    if scene_id == "L3":
        adjusted.pop("upper_body_garment", None)

    if scene_id == "I1" and upper_is_outer and isinstance(upper, dict):
        region_preserved = adjusted.setdefault("upper_body_garment", {})
        layering = upper.get("layering_structure") or {}
        if layering.get("is_outerwear") is True:
            region_preserved["layering_structure"] = {
                key: value
                for key, value in layering.items()
                if value not in (None, "not_applicable", "not_visible")
            }

    return adjusted
