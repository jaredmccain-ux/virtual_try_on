"""Focused Rivo prompt for supplemental fields on existing paired annotations."""

from __future__ import annotations

import json
from typing import Any

EMBELLISHMENT_VALUES = [
    "none",
    "ruffle",
    "pleat",
    "sequin",
    "embroidery",
    "lace_trim",
    "fringe",
    "bow",
    "button_detail",
    "zipper_detail",
    "distressed",
    "cutout",
    "other",
]

GENDER_VALUES = ["male", "female"]
HAVE_INNER_VALUES = ["true", "false", "not_applicable"]

SYSTEM_PROMPT = f"""You are an image interpretation assistant.

You will see two images:
- Image 1: model/person image
- Image 2: garment image

You will also receive the existing JSON annotation for this pair.
Your task is ONLY to add the missing supplemental fields listed below.
Do NOT change any existing field values. Do NOT re-annotate the whole sample.

## Fields to output

1. garment_gender
   - Judge whether Image 2 (the garment image) is menswear or womenswear.
   - Values: {" / ".join(GENDER_VALUES)}
   - male = menswear; female = womenswear

2. person_regions
   - For each person region that already has is_present=true in the provided annotation,
     output embellishment for that region's visible garment.
   - Region keys must be exactly:
     upper_body_garment / lower_body_garment / whole_body_garment
   - Skip regions with is_present=false (do not include them).

   embellishment:
   - Decorative details on the garment worn in that region on Image 1.
   - Values: {" / ".join(EMBELLISHMENT_VALUES)}
   - Use none when no obvious decorative detail is visible.

   have_inner (ONLY for upper_body_garment and whole_body_garment):
   - Whether a separate inner layer is visibly worn under the outer garment in Image 1.
   - Output have_inner ONLY when the provided annotation for that region has layering_structure.is_outerwear=true.
   - If is_outerwear is not true for that region, set have_inner=not_applicable.
   - When is_outerwear=true:
     - true = a distinct inner top/shirt/dress layer is visible under the outerwear
     - false = outerwear is worn without a visible inner layer
   - Values: true / false / not_applicable

## Output format

Output exactly one JSON object, no markdown, no comments:

{{
  "garment_gender": "female",
  "person_regions": {{
    "upper_body_garment": {{
      "embellishment": "none",
      "have_inner": "not_applicable"
    }},
    "lower_body_garment": {{
      "embellishment": "none"
    }}
  }}
}}

Rules:
- Do not add keys other than garment_gender and person_regions.
- person_regions values only contain embellishment, and optionally have_inner.
- lower_body_garment entries must NOT contain have_inner.
- Use the existing annotation to decide is_outerwear before filling have_inner.
- If unsure about embellishment, prefer none rather than guessing.
"""


def build_user_prompt(*, source_sample_id: str, existing_annotation: dict[str, Any]) -> str:
    context_json = json.dumps(existing_annotation, ensure_ascii=False, indent=2)
    return (
        f"source_sample_id: {source_sample_id}\n"
        "Existing annotation JSON (read-only context):\n"
        f"{context_json}\n\n"
        "Look at Image 1 and Image 2, then output ONLY the supplemental JSON object "
        "defined in the system prompt."
    )


def build_output_schema_hint() -> dict[str, Any]:
    return {
        "garment_gender": "female",
        "person_regions": {
            "upper_body_garment": {
                "embellishment": "none",
                "have_inner": "not_applicable",
            },
            "lower_body_garment": {"embellishment": "none"},
        },
    }
