"""
GPT-based evaluation for demo-generated virtual try-on images.

Evaluates each generated image on two metrics (CLIP disabled):
  1. Edit Correctness  — are editing requirements correctly applied?
  2. Preservation      — are preservation requirements satisfied?

Usage (called by app.py):
    from score_eval_demo import evaluate
    result = evaluate(person_img, garment_img, result_img, instruction)
"""


import base64
import io
import json
import os
import re
import time

from openai import OpenAI
from PIL import Image

# ════════════════════════════════════════════════════════════════════
# ★ 在此填入你的 API Key
# ════════════════════════════════════════════════════════════════════
API_KEY = "YOUR_API_KEY_HERE"
API_BASE_URL = "https://api.rivoapi.com/v1"
MODEL_NAME = "gpt-5.4"
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ════════════════════════════════════════════════════════════════════
# CLIP scoring (disabled for demo)
# ════════════════════════════════════════════════════════════════════
# import torch
# from transformers import CLIPModel, CLIPProcessor
# CLIP_MODEL_PATH = "/data2/dingxin/memgen/pretrained_models/clip-vit-large-patch14-336"
#
# def compute_clip_score(image_path, text):
#     ...  # See score_eval.py for full implementation


# ── Instruction Parsing ────────────────────────────────────────────

def parse_instruction(instruction):
    """Parse structured instruction into edits and preserves."""
    result = {
        "task": "",
        "edits": [],
        "preserves": [],
        "target_body_part": None,
    }

    tag_pattern = re.compile(r'\[(EDIT|PRESERVE)\s*·\s*(\w+)\]')
    parts = tag_pattern.split(instruction)

    if parts:
        result["task"] = parts[0].strip().rstrip(".")

    for m in tag_pattern.finditer(instruction):
        tag_type = m.group(1)
        category = m.group(2)

        start = m.end()
        next_tag = tag_pattern.search(instruction, start)
        content = instruction[start:next_tag.start()].strip() if next_tag else instruction[start:].strip()

        if tag_type == "EDIT":
            result["edits"].append({
                "category": category,
                "description": content,
            })
        else:
            cleaned = re.sub(r'^(Preserve|Keep)\s+', '', content).rstrip('.')
            items = [item.strip() for item in re.split(r',|and', cleaned) if item.strip()]
            items = [re.sub(r'\s*unchanged$', '', item).strip() for item in items]
            items = [item for item in items if item]

            result["preserves"].append({
                "category": category,
                "description": content,
                "items": items,
            })

    task_lower = result["task"].lower()
    if any(k in task_lower for k in ("upper-body", "upper_body", "upper body", "outerwear")):
        result["target_body_part"] = "upper_body"
    elif any(k in task_lower for k in ("lower-body", "lower_body", "lower body")):
        result["target_body_part"] = "lower_body"
    elif any(k in task_lower for k in ("dress", "full-body", "one-piece", "outfit")):
        result["target_body_part"] = "full_body"

    if result["target_body_part"] == "upper_body":
        result["preserves"].append({
            "category": "cross_body",
            "description": "Lower-body clothing must remain unchanged.",
            "items": ["lower_body_clothing", "footwear"],
        })
    elif result["target_body_part"] == "lower_body":
        result["preserves"].append({
            "category": "cross_body",
            "description": "Upper-body clothing must remain unchanged.",
            "items": ["upper_body_clothing"],
        })

    # ── Fallback: plain-text instructions (no [EDIT] / [PRESERVE] tags) ──
    if not result["edits"]:
        # Extract "Keep ..." clauses as preserves, remainder as the edit
        keep_pattern = re.compile(
            r'[,.]?\s*(Keep|Preserve|Maintain)\s+(.+?)(?=[,.]\s*(?:Keep|Preserve|Maintain)|\.$|$)',
            re.IGNORECASE,
        )
        keeps = keep_pattern.findall(instruction)
        # Remove keep-clause from the edit description
        edit_text = keep_pattern.sub('', instruction).strip().rstrip('.')
        if edit_text:
            result["edits"].append({
                "category": "instruction",
                "description": edit_text,
            })
        for _, clause in keeps:
            result["preserves"].append({
                "category": "keep_clause",
                "description": clause.strip().rstrip('.'),
                "items": [],
            })
        # If still no preserves at all, add generic ones
        if not result["preserves"]:
            result["preserves"].append({
                "category": "identity",
                "description": "Person identity (face, body shape) must remain unchanged.",
                "items": ["face", "body_shape"],
            })

    return result


# ── Image Handling ─────────────────────────────────────────────────

def encode_image(img, max_size=1024) -> str:
    """Encode a PIL Image to base64 data URL, resizing if needed."""
    # Gradio may pass a tuple (image, mask) — extract the image
    if isinstance(img, tuple):
        img = img[0]
    if img is None:
        return ""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── GPT Prompt ─────────────────────────────────────────────────────

def build_prompt(parsed):
    """Build the GPT evaluation prompt (text-only, images added later)."""
    edit_lines = []
    for i, e in enumerate(parsed["edits"], 1):
        edit_lines.append(f"{i}. [{e['category']}] {e['description']}")

    preserve_lines = []
    for i, p in enumerate(parsed["preserves"], 1):
        preserve_lines.append(f"{i}. [{p['category']}] {p['description']}")

    edits_text = "\n".join(edit_lines) if edit_lines else "(none)"
    preserves_text = "\n".join(preserve_lines) if preserve_lines else "(none)"

    edit_schema = {}
    for e in parsed["edits"]:
        edit_schema[e["category"]] = {
            "judgment": "yes / partial / no",
            "reason": "brief justification in English"
        }

    preserve_schema = {}
    for p in parsed["preserves"]:
        preserve_schema[p["category"]] = {
            "judgment": "yes / partial / no",
            "reason": "brief justification in English"
        }

    json_schema = {
        "edit_scores": edit_schema,
        "preserve_scores": preserve_schema,
        "overall_comment": "brief overall assessment in 1-2 sentences"
    }

    system_prompt = """You are an expert evaluator for AI-generated fashion image editing.
Your task is to compare three images and judge whether the editing instructions were correctly followed.

─── THE EDITING PROCESS (critical to understand) ───

The pipeline works in two conceptual steps:
  Step A: The person (Image 1) puts on the reference garment (Image 2).
  Step B: The [EDIT] instructions are applied to modify specific attributes of that garment on the person.

So the generated result (Image 3) should show:
  - The person from Image 1
  - Wearing the garment from Image 2
  - With [EDIT] changes applied to the garment

─── THREE CATEGORIES OF ATTRIBUTES (different comparison baselines) ───

CATEGORY 1 — Person identity & scene:
  identity, face, pose, hair, body shape, background, lighting
  → Compare to Image 1 (Source Person)
  → These must NOT change from Image 1

CATEGORY 2 — Garment intrinsic attributes (what the garment IS):
  neckline, sleeve_length, sleeve_type, collar_type, hem_length, hem_shape,
  color, pattern, material_texture, embellishment
  → Compare to Image 2 (Reference Garment)
  → This is the NEW garment being worn. Its attributes come from Image 2.
  → [EDIT] on these: the result should differ from Image 2 in the specified way
    (e.g., if Image 2 is white and [EDIT] says "color from white to teal",
    the result MUST be teal — this is correct, not a mistake)
  → [PRESERVE] on these: the result MUST match Image 2

CATEGORY 3 — Wearing style (HOW the garment is worn, not what it is):
  tucking_style, sleeve_state, pants_cuff_state
  → These describe how the garment is worn on the body (tucked/untucked, sleeves rolled/down).
  → [EDIT] changes the wearing style as specified.
  → [PRESERVE] means the wearing style should remain as shown on the source person (Image 1).

─── SCORING GUIDELINES ───
- "yes" = the requirement is fully and clearly satisfied
- "partial" = the requirement is partially satisfied or ambiguous
- "no" = the requirement is clearly not satisfied

Be objective and specific. Always identify which category an attribute belongs to,
then compare against the correct baseline image.

IMPORTANT: Return ONLY valid JSON, no other text."""

    user_prompt = f"""Evaluate this fashion image editing result.

**Task**: {parsed['task']}

**Image 1 (Source Person)**: The original person — baseline for: person identity/face/pose/hair/background, original wearing style (tucking, sleeve_state).
**Image 2 (Reference Garment)**: The garment that was put onto the person — baseline for garment attributes (neckline, sleeve length, color, pattern, material, etc.).
**Image 3 (Generated Result)**: The edited output image to evaluate.

---
HOW TO EVALUATE:

**Metric 1 — Edit Correctness**
For each [EDIT] requirement below, check whether the specified change was correctly applied.
KEY RULE: [EDIT] describes intentional changes. If Image 2 is white and [EDIT] says "color: white → teal", then a TEAL result is CORRECT.
{edits_text}

→ Compare the Generated Result to Image 2 (Reference Garment), then check if the [EDIT] change is correctly applied.

---

**Metric 2 — Content Preservation**
For each [PRESERVE] requirement below, judge whether the content was kept as required.
When judging, use the CORRECT baseline for each attribute type:
  - Person attributes (identity, face, pose, hair, body shape, background, lighting)
    → compare to Image 1 (Source Person)
  - Garment attributes (neckline, sleeve length, sleeve type, collar, color, pattern,
    material, embellishment, hem, etc.) → compare to Image 2 (Reference Garment)
  - Wearing style (tucking, sleeve_state, pants cuff) → compare to Image 1 (how it was worn)

{preserves_text}

---

Return your evaluation as JSON with this EXACT structure:
```json
{json.dumps(json_schema, indent=2, ensure_ascii=False)}
```
"""

    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
            ],
        },
    ]


def add_images_to_message(messages, person_img, garment_img, result_img):
    """Add three images to the user message as base64 data URLs."""
    images_data = [
        ("Source Person", person_img),
        ("Reference Garment", garment_img),
        ("Generated Result", result_img),
    ]

    user_content = messages[1]["content"]
    for label, img in images_data:
        b64 = encode_image(img)
        user_content.append({"type": "text", "text": f"\n\n**{label}**:"})
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": "auto",
            },
        })


# ── GPT API Call ───────────────────────────────────────────────────

def call_gpt(messages, max_tokens=2048):
    """Call GPT API with retry logic. Returns parsed JSON or None."""
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.1,
            )
            content = response.choices[0].message.content.strip()

            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)

            return json.loads(content)

        except json.JSONDecodeError as e:
            print(f"  [WARN] JSON parse error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [WARN] API error (attempt {attempt}): {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
            else:
                print(f"  [ERROR] All {MAX_RETRIES} attempts failed")
                return None

    return None


# ── Formatting ─────────────────────────────────────────────────────

def _badge(judgment):
    """Return a colored badge for the judgment."""
    j = judgment.lower()
    if j == "yes":
        return "✅ yes"
    elif j == "partial":
        return "⚠️ partial"
    else:
        return "❌ no"


def format_result(result, elapsed_seconds=0):
    """Format GPT result dict into a Gradio-friendly Markdown string."""
    if result is None:
        return "**Evaluation failed.** Please check your API key and try again."

    lines = []

    # Edit correctness
    edit_scores = result.get("edit_scores", {})
    edit_yes = sum(1 for v in edit_scores.values() if v.get("judgment", "").lower() == "yes")
    edit_total = len(edit_scores)
    lines.append(f"### Edit Correctness  {edit_yes}/{edit_total}")
    lines.append("")
    if edit_scores:
        for cat, item in edit_scores.items():
            j = _badge(item.get("judgment", "no"))
            reason = item.get("reason", "")
            lines.append(f"- **{cat}** — {j}")
            if reason:
                lines.append(f"  {reason}")
    else:
        lines.append("_No edit requirements detected._")
    lines.append("")

    # Preservation
    preserve_scores = result.get("preserve_scores", {})
    pres_yes = sum(1 for v in preserve_scores.values() if v.get("judgment", "").lower() == "yes")
    pres_total = len(preserve_scores)
    lines.append(f"### Content Preservation  {pres_yes}/{pres_total}")
    lines.append("")
    if preserve_scores:
        for cat, item in preserve_scores.items():
            j = _badge(item.get("judgment", "no"))
            reason = item.get("reason", "")
            lines.append(f"- **{cat}** — {j}")
            if reason:
                lines.append(f"  {reason}")
    else:
        lines.append("_No preservation requirements detected._")
    lines.append("")

    # Overall comment
    comment = result.get("overall_comment", "")
    if comment:
        lines.append(f"### Summary")
        lines.append("")
        lines.append(comment)

    if elapsed_seconds > 0:
        lines.append(f"\n---\n*Evaluated in {elapsed_seconds:.1f}s*")

    return "\n".join(lines)


# ── Public API (called by app.py) ──────────────────────────────────

def evaluate(person_img, garment_img, result_img, instruction, api_key=None):
    """
    Evaluate a generated try-on image.

    Args:
        person_img: PIL.Image — source person image
        garment_img: PIL.Image — reference garment image
        result_img: PIL.Image — generated result image
        instruction: str — editing instruction text
        api_key: str — override API key (optional)

    Returns:
        str: formatted Markdown evaluation result
    """
    if api_key and api_key.strip() and api_key.strip() != "YOUR_API_KEY_HERE":
        global API_KEY
        API_KEY = api_key.strip()

    if API_KEY == "YOUR_API_KEY_HERE":
        return "**Error:** Please set API_KEY in `score_eval_demo.py` first."

    if person_img is None or garment_img is None or result_img is None:
        return "**Error:** All three images (person, garment, result) are required."

    if not instruction or not instruction.strip():
        return "**Error:** Editing instruction is required."

    # Gallery returns a list; take the first image
    if isinstance(result_img, list):
        result_img = result_img[0] if result_img else None
    if result_img is None:
        return "**Error:** No generated result to evaluate. Run Generate first."

    # Parse instruction
    parsed = parse_instruction(instruction)

    # Build prompt
    messages = build_prompt(parsed)

    # Add images
    add_images_to_message(messages, person_img, garment_img, result_img)

    # Call GPT
    t0 = time.time()
    result = call_gpt(messages)
    elapsed = time.time() - t0

    # Format and return
    return format_result(result, elapsed)
