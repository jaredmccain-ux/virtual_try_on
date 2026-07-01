#!/usr/bin/env python3
"""Annotate person + garment image pairs via Agnes, Rivo, or Volcengine Ark APIs."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import socket
import sys
import time
import urllib.error
import urllib.request
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Callable

from prompt_en import SYSTEM_PROMPT, build_user_prompt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path("/data1/virtual_tryon")
DEFAULT_DATASET_ROOT = Path("/data1/virtual_tryon/Datasets/eval_firsttest")
DEFAULT_INPUT_JSONL = Path(
    "/data1/virtual_tryon/annoted-code/annotations_step01_base.jsonl"
)

PROVIDER_AGNES = "agnes"
PROVIDER_RIVO = "rivo"
PROVIDER_ARK = "ark"

OPENAI_COMPATIBLE_PROVIDERS = {PROVIDER_AGNES, PROVIDER_RIVO}

AGNES_DEFAULT_BASE_URL = "https://apihub.agnes-ai.com/v1"
AGNES_DEFAULT_MODEL = "agnes-2.0-flash"

RIVO_DEFAULT_BASE_URL = "https://api.rivoapi.com/v1"
RIVO_DEFAULT_MODEL = "gpt-5.4"

ARK_DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
ARK_DEFAULT_MODEL = "doubao-seed-2-0-mini-260428"

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    PROVIDER_AGNES: {
        "base_url": AGNES_DEFAULT_BASE_URL,
        "model": AGNES_DEFAULT_MODEL,
        "api_key_env": "AGNES_API_KEY",
        "base_url_env": "AGNES_BASE_URL",
        "model_env": "AGNES_MODEL",
    },
    PROVIDER_RIVO: {
        "base_url": RIVO_DEFAULT_BASE_URL,
        "model": RIVO_DEFAULT_MODEL,
        "api_key_env": "RIVO_API_KEY",
        "base_url_env": "RIVO_BASE_URL",
        "model_env": "RIVO_MODEL",
    },
    PROVIDER_ARK: {
        "base_url": ARK_DEFAULT_BASE_URL,
        "model": ARK_DEFAULT_MODEL,
        "api_key_env": "ARK_API_KEY",
        "base_url_env": "ARK_BASE_URL",
        "model_env": "ARK_MODEL",
    },
}

RAW_OUTPUT_DIR = SCRIPT_DIR / "model_rawsay"
PARSED_OUTPUT_JSONL = SCRIPT_DIR / "annotations_api.jsonl"
FAILURE_LOG_JSONL = SCRIPT_DIR / "logs" / "annotate_api_failures.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate image pairs via Agnes, Rivo, or Volcengine Ark API."
    )
    parser.add_argument(
        "--provider",
        choices=[PROVIDER_AGNES, PROVIDER_RIVO, PROVIDER_ARK],
        default=os.environ.get("ANNOTATE_PROVIDER", PROVIDER_AGNES),
        help=(
            "API provider: agnes / rivo (OpenAI-compatible chat/completions) "
            "or ark (Doubao /responses)."
        ),
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        default=DEFAULT_INPUT_JSONL,
        help="JSONL with person_image.file_name and garment_image.file_name.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root for resolving image paths.",
    )
    parser.add_argument("--base-url", default="", help="API base URL.")
    parser.add_argument("--api-key", default="", help="API key.")
    parser.add_argument("--model", default="", help="Model ID.")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum completion / output tokens.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--retry",
        type=int,
        default=2,
        help="Retries after request/parse failure.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N samples.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip samples that already have model_rawsay output for this model.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build requests only; do not call API or write outputs.",
    )
    parser.add_argument(
        "--raw-output-file",
        type=Path,
        default=None,
        help="Single TXT file for all raw model outputs (default: model_rawsay/{provider}__{model}.txt).",
    )
    parser.add_argument(
        "--parsed-output-jsonl",
        type=Path,
        default=None,
        help="JSONL for parsed model JSON plus metadata (default: annotations_{provider}.jsonl).",
    )
    parser.add_argument(
        "--failure-log-jsonl",
        type=Path,
        default=None,
        help="Failure log path (default: logs/annotate_{provider}_failures.jsonl).",
    )
    args = parser.parse_args()

    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.retry < 0:
        parser.error("--retry must be non-negative")

    provider_defaults = PROVIDER_DEFAULTS[args.provider]
    args.base_url = args.base_url or os.environ.get(
        provider_defaults["base_url_env"], provider_defaults["base_url"]
    )
    args.api_key = args.api_key or os.environ.get(provider_defaults["api_key_env"], "")
    args.model = args.model or os.environ.get(
        provider_defaults["model_env"], provider_defaults["model"]
    )
    if not args.dry_run and not args.api_key:
        parser.error(
            f"{args.provider} API key required. Set "
            f"{provider_defaults['api_key_env']} or pass --api-key."
        )

    safe_model = sanitize_for_filename(args.model)
    if args.raw_output_file is None:
        args.raw_output_file = RAW_OUTPUT_DIR / f"{args.provider}__{safe_model}.txt"
    if args.parsed_output_jsonl is None:
        args.parsed_output_jsonl = SCRIPT_DIR / f"annotations_{args.provider}.jsonl"
    if args.failure_log_jsonl is None:
        args.failure_log_jsonl = (
            SCRIPT_DIR / "logs" / f"annotate_{args.provider}_failures.jsonl"
        )

    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                rows.append(json.loads(text))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_no}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def image_path_to_data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def resolve_image_path(dataset_root: Path, file_name: str) -> Path:
    path = Path(file_name)
    if path.is_absolute():
        return path
    return dataset_root / path


def sanitize_for_filename(value: str) -> str:
    return re.sub(r"[^\w.\-]+", "_", value)


@contextmanager
def force_ipv4_dns():
    """Prefer IPv4 when the host has broken IPv6 connectivity."""
    original_getaddrinfo = socket.getaddrinfo

    def ipv4_getaddrinfo(
        host: str,
        port: str | int | None,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ):
        del family
        return original_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    socket.getaddrinfo = ipv4_getaddrinfo
    try:
        yield
    finally:
        socket.getaddrinfo = original_getaddrinfo


def post_json(
    *,
    url: str,
    api_key: str,
    payload: dict[str, Any],
    timeout: float,
    use_ipv4: bool,
) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    try:
        if use_ipv4:
            context = force_ipv4_dns()
        else:
            context = nullcontext()
        with context, urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {error_body}") from exc


def chat_completions_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/chat/completions"):
        return normalized
    return f"{normalized}/chat/completions"


def responses_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    if normalized.endswith("/responses"):
        return normalized
    return f"{normalized}/responses"


def build_agnes_messages(
    *,
    person_image_path: Path,
    garment_image_path: Path,
    source_sample_id: str,
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Image 1: model/person image"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(person_image_path)},
                },
                {"type": "text", "text": "Image 2: garment image"},
                {
                    "type": "image_url",
                    "image_url": {"url": image_path_to_data_url(garment_image_path)},
                },
                {"type": "text", "text": build_user_prompt(source_sample_id)},
            ],
        },
    ]


def build_ark_input(
    *,
    person_image_path: Path,
    garment_image_path: Path,
    source_sample_id: str,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "input_text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": "Image 1: model/person image"},
                {
                    "type": "input_image",
                    "image_url": image_path_to_data_url(person_image_path),
                },
                {"type": "input_text", "text": "Image 2: garment image"},
                {
                    "type": "input_image",
                    "image_url": image_path_to_data_url(garment_image_path),
                },
                {"type": "input_text", "text": build_user_prompt(source_sample_id)},
            ],
        },
    ]


def call_openai_chat_completions(
    *,
    messages: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: float,
    use_ipv4: bool,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    response_payload = post_json(
        url=chat_completions_url(base_url),
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        use_ipv4=use_ipv4,
    )

    choices = response_payload.get("choices")
    if not choices:
        raise ValueError(f"response has no choices: {response_payload}")

    content = choices[0].get("message", {}).get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "".join(parts)
    if not isinstance(content, str):
        raise ValueError(f"response content is not text: {content!r}")
    return content


def extract_ark_response_text(response_payload: dict[str, Any]) -> str:
    output = response_payload.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            if item.get("type") != "message":
                continue
            content = item.get("content")
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text")
                        if isinstance(text, str):
                            parts.append(text)
                if parts:
                    return "".join(parts)
            if isinstance(content, str):
                return content

    for key in ("output_text", "text"):
        value = response_payload.get(key)
        if isinstance(value, str) and value.strip():
            return value

    choices = response_payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content

    raise ValueError(f"unable to extract text from Ark response: {response_payload}")


def call_ark_responses(
    *,
    input_messages: list[dict[str, Any]],
    base_url: str,
    api_key: str,
    model: str,
    max_tokens: int,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "input": input_messages,
        "max_output_tokens": max_tokens,
        "thinking": {"type": "disabled"},
    }
    response_payload = post_json(
        url=responses_url(base_url),
        api_key=api_key,
        payload=payload,
        timeout=timeout,
        use_ipv4=False,
    )
    return extract_ark_response_text(response_payload)


def extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    raise ValueError("no JSON object found in model output")


def format_raw_output_text(raw_text: str, parsed: dict[str, Any] | None = None) -> str:
    """Save model output as vertically indented JSON when possible."""
    obj = parsed
    if obj is None:
        try:
            obj = extract_json_object(raw_text)
        except ValueError:
            return raw_text.rstrip() + "\n"
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


RAW_RECORD_SEPARATOR = "=" * 80


def format_raw_record(
    *,
    source_sample_id: str,
    model_id: str,
    person_image: str,
    garment_image: str,
    raw_text: str,
    parsed: dict[str, Any] | None,
) -> str:
    """One labeled block per (sample, model) pair inside the shared raw TXT file."""
    header = (
        f"{RAW_RECORD_SEPARATOR}\n"
        f"PAIR source_sample_id={source_sample_id}\n"
        f"model_id={model_id}\n"
        f"person_image={person_image}\n"
        f"garment_image={garment_image}\n"
        f"{RAW_RECORD_SEPARATOR}\n"
    )
    return header + format_raw_output_text(raw_text, parsed) + "\n"


def append_raw_record(raw_file: Path, record_text: str) -> None:
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_file.open("a", encoding="utf-8") as handle:
        handle.write(record_text)


def scan_existing_sample_ids(raw_file: Path) -> set[str]:
    if not raw_file.is_file():
        return set()
    ids: set[str] = set()
    for line in raw_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"PAIR source_sample_id=(\S+)", line.strip())
        if match:
            ids.add(match.group(1))
    return ids


def parse_raw_records(path: Path) -> list[dict[str, Any]]:
    """Parse single-file raw output from format_raw_record into row dicts."""
    if not path.is_file():
        return []

    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    sep = RAW_RECORD_SEPARATOR
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line == sep:
            i += 1
            continue
        if not line.startswith("PAIR source_sample_id="):
            i += 1
            continue

        meta: dict[str, str] = {"source_sample_id": line.split("=", 1)[1].strip()}
        i += 1
        while i < len(lines) and lines[i].strip() != sep:
            row_line = lines[i].strip()
            if row_line.startswith("model_id="):
                meta["model_id"] = row_line.split("=", 1)[1].strip()
            elif row_line.startswith("person_image="):
                meta["person_image"] = row_line.split("=", 1)[1].strip()
            elif row_line.startswith("garment_image="):
                meta["garment_image"] = row_line.split("=", 1)[1].strip()
            i += 1
        if i < len(lines) and lines[i].strip() == sep:
            i += 1

        json_lines: list[str] = []
        while i < len(lines):
            row_line = lines[i]
            stripped = row_line.strip()
            if stripped == sep or stripped.startswith("PAIR source_sample_id="):
                break
            json_lines.append(row_line)
            i += 1

        json_text = "\n".join(json_lines).strip()
        if not json_text:
            continue
        try:
            annotation = json.loads(json_text)
        except json.JSONDecodeError:
            continue
        rows.append(
            {
                "source_sample_id": meta["source_sample_id"],
                "model_id": meta.get("model_id", ""),
                "person_image": meta.get("person_image", ""),
                "garment_image": meta.get("garment_image", ""),
                "annotation": annotation,
                "status": "ok",
            }
        )
    return rows


def build_result_metadata(
    *,
    source_sample_id: str,
    provider: str,
    model_id: str,
    raw_output_file: str,
    annotation: dict[str, Any],
    status: str,
    person_image: str = "",
    garment_image: str = "",
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    row = {
        "source_sample_id": source_sample_id,
        "provider": provider,
        "model_id": model_id,
        "model": model_id,
        "person_image": person_image,
        "garment_image": garment_image,
        "raw_output_file": raw_output_file,
        "annotation": annotation,
        "status": status,
    }
    if elapsed_seconds is not None:
        row["elapsed_seconds"] = elapsed_seconds
    return row


def annotate_sample(
    sample: dict[str, Any],
    *,
    provider: str,
    dataset_root: Path,
    base_url: str,
    api_key: str,
    model_id: str,
    max_tokens: int,
    timeout: float,
    retry: int,
) -> tuple[str, dict[str, Any]]:
    source_sample_id = str(sample.get("source_sample_id", "unknown"))
    person_file = sample["person_image"]["file_name"]
    garment_file = sample["garment_image"]["file_name"]
    person_path = resolve_image_path(dataset_root, person_file)
    garment_path = resolve_image_path(dataset_root, garment_file)
    if not person_path.is_file():
        raise FileNotFoundError(f"person image not found: {person_path}")
    if not garment_path.is_file():
        raise FileNotFoundError(f"garment image not found: {garment_path}")

    if provider == PROVIDER_ARK:
        raw_text = _call_with_retry(
            lambda: call_ark_responses(
                input_messages=build_ark_input(
                    person_image_path=person_path,
                    garment_image_path=garment_path,
                    source_sample_id=source_sample_id,
                ),
                base_url=base_url,
                api_key=api_key,
                model=model_id,
                max_tokens=max_tokens,
                timeout=timeout,
            ),
            retry=retry,
        )
    elif provider in OPENAI_COMPATIBLE_PROVIDERS:
        raw_text = _call_with_retry(
            lambda: call_openai_chat_completions(
                messages=build_agnes_messages(
                    person_image_path=person_path,
                    garment_image_path=garment_path,
                    source_sample_id=source_sample_id,
                ),
                base_url=base_url,
                api_key=api_key,
                model=model_id,
                max_tokens=max_tokens,
                timeout=timeout,
                use_ipv4=provider == PROVIDER_AGNES,
            ),
            retry=retry,
        )
    else:
        raise ValueError(f"unsupported provider: {provider}")
    parsed = extract_json_object(raw_text)
    return raw_text, parsed


def _call_with_retry(call: Callable[[], str], *, retry: int) -> str:
    last_error: Exception | None = None
    for attempt in range(retry + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - retry loop
            last_error = exc
            if attempt < retry:
                time.sleep(min(2 ** attempt, 8))
    assert last_error is not None
    raise last_error


def main() -> int:
    args = parse_args()
    samples = read_jsonl(args.input_jsonl)
    if args.limit is not None:
        samples = samples[: args.limit]

    raw_file: Path = args.raw_output_file
    raw_file.parent.mkdir(parents=True, exist_ok=True)
    parsed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    def rel_to_project(path: Path) -> str:
        try:
            return str(path.relative_to(PROJECT_ROOT))
        except ValueError:
            return str(path)

    existing_ids: set[str] = set()
    if args.skip_existing:
        existing_ids = scan_existing_sample_ids(raw_file)
    elif raw_file.is_file():
        raw_file.unlink()

    print(f"provider={args.provider}", flush=True)
    print(f"input_jsonl={args.input_jsonl}", flush=True)
    print(f"samples={len(samples)}", flush=True)
    print(f"model_id={args.model}", flush=True)
    print(f"base_url={args.base_url}", flush=True)
    print(f"raw_output_file={raw_file}", flush=True)

    if args.dry_run:
        for index, sample in enumerate(samples, start=1):
            source_sample_id = str(sample.get("source_sample_id", "unknown"))
            person_path = resolve_image_path(
                args.dataset_root, sample["person_image"]["file_name"]
            )
            garment_path = resolve_image_path(
                args.dataset_root, sample["garment_image"]["file_name"]
            )
            print(
                f"[dry-run] {index}/{len(samples)} "
                f"provider={args.provider} model_id={args.model} "
                f"source_sample_id={source_sample_id} "
                f"person={person_path} garment={garment_path} "
                f"raw_output_file={raw_file}",
                flush=True,
            )
        print("dry_run=true", flush=True)
        return 0

    started = time.perf_counter()
    for index, sample in enumerate(samples, start=1):
        source_sample_id = str(sample.get("source_sample_id", "unknown"))
        person_rel = rel_to_project(
            resolve_image_path(args.dataset_root, sample["person_image"]["file_name"])
        )
        garment_rel = rel_to_project(
            resolve_image_path(args.dataset_root, sample["garment_image"]["file_name"])
        )

        if args.skip_existing and source_sample_id in existing_ids:
            print(
                f"[skip] {index}/{len(samples)} source_sample_id={source_sample_id} "
                f"model_id={args.model}",
                flush=True,
            )
            continue

        sample_started = time.perf_counter()
        print(
            f"[call] {index}/{len(samples)} provider={args.provider} "
            f"model_id={args.model} source_sample_id={source_sample_id} "
            "uploading images and waiting for API response...",
            flush=True,
        )
        try:
            raw_text, parsed = annotate_sample(
                sample,
                provider=args.provider,
                dataset_root=args.dataset_root,
                base_url=args.base_url,
                api_key=args.api_key,
                model_id=args.model,
                max_tokens=args.max_tokens,
                timeout=args.timeout,
                retry=args.retry,
            )
            append_raw_record(
                raw_file,
                format_raw_record(
                    source_sample_id=source_sample_id,
                    model_id=args.model,
                    person_image=person_rel,
                    garment_image=garment_rel,
                    raw_text=raw_text,
                    parsed=parsed,
                ),
            )
            existing_ids.add(source_sample_id)
            parsed_rows.append(
                build_result_metadata(
                    source_sample_id=source_sample_id,
                    provider=args.provider,
                    model_id=args.model,
                    raw_output_file=str(raw_file),
                    annotation=parsed,
                    status="ok",
                    person_image=person_rel,
                    garment_image=garment_rel,
                    elapsed_seconds=round(time.perf_counter() - sample_started, 2),
                )
            )
            print(
                f"[ok] {index}/{len(samples)} source_sample_id={source_sample_id} "
                f"model_id={args.model} "
                f"elapsed={time.perf_counter() - sample_started:.2f}s",
                flush=True,
            )
        except Exception as exc:  # noqa: BLE001
            failure = {
                "source_sample_id": source_sample_id,
                "provider": args.provider,
                "model_id": args.model,
                "error": str(exc),
                "elapsed_seconds": round(time.perf_counter() - sample_started, 2),
            }
            failures.append(failure)
            print(
                f"[fail] {index}/{len(samples)} source_sample_id={source_sample_id} "
                f"model_id={args.model} error={exc}",
                flush=True,
            )

    write_jsonl(args.parsed_output_jsonl, parsed_rows)
    if failures:
        write_jsonl(args.failure_log_jsonl, failures)

    try:
        from build_review_manifest import main as build_review_manifest_main

        build_review_manifest_main()
    except Exception as exc:  # noqa: BLE001
        print(f"review_manifest_build_failed={exc}", flush=True)

    try:
        from build_review_rivo_manifest import main as build_review_rivo_manifest_main

        build_review_rivo_manifest_main()
    except Exception as exc:  # noqa: BLE001
        print(f"review_rivo_manifest_build_failed={exc}", flush=True)

    print(f"processed_ok={len(parsed_rows)}", flush=True)
    print(f"failed={len(failures)}", flush=True)
    print(f"parsed_output_jsonl={args.parsed_output_jsonl}", flush=True)
    print(f"total_seconds={time.perf_counter() - started:.2f}", flush=True)
    if failures:
        print(f"failure_log_jsonl={args.failure_log_jsonl}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
