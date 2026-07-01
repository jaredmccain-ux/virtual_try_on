#!/usr/bin/env python3
"""Build a self-contained share bundle with HTML viewer and local images."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
DATA_DIR = ROOT_DIR / "data"
PROJECT_ROOT = ROOT_DIR.parent.parent
DEFAULT_OUTPUT = WEB_DIR / "share_bundle"
DEFAULT_ZIP = ROOT_DIR / "constrcut_instruction_share.zip"

DIMENSIONS = (
    "wearing_style",
    "fit_silhouette",
    "local_structure",
    "appearance",
    "layering_structure",
)

DATA_FILES = (
    "annotations_paired_with_edit.jsonl",
    "annotations_unpair_with_edit.jsonl",
    "testset_paired.jsonl",
    "testset_unpair.jsonl",
    "paired_edit_type_summary.csv",
    "unpair_edit_type_summary.csv",
    "README.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build HTML share bundle.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--zip", type=Path, default=DEFAULT_ZIP)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--skip-zip", action="store_true")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                rows.append(json.loads(text))
    return rows


def flatten_changes(changes_by_dimension: dict[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for dimension in DIMENSIONS:
        attrs = changes_by_dimension.get(dimension) or {}
        if not isinstance(attrs, dict):
            continue
        for attribute, change in attrs.items():
            if not isinstance(change, dict):
                continue
            items.append(
                {
                    "dimension": dimension,
                    "attribute": attribute,
                    "from": change.get("from"),
                    "to": change.get("to"),
                }
            )
    return items


def to_viewer_row(row: dict[str, Any], *, project_root: Path, images_dir: Path) -> dict[str, Any]:
    edit_task = row.get("edit_task") or {}
    instruction = row.get("instruction") or {}
    changes = edit_task.get("changes_by_dimension") or edit_task.get("edit_spec") or {}

    person_rel = str(row.get("person_image") or "")
    garment_rel = str(row.get("garment_image") or "")

    def copy_image(rel_path: str) -> str:
        if not rel_path:
            return ""
        src = project_root / rel_path
        if not src.is_file():
            raise FileNotFoundError(f"image not found: {src}")
        dst = images_dir / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        return f"images/{rel_path}"

    source_attrs = row.get("source_attributes") or {}
    if isinstance(source_attrs.get("whole_body_garment"), dict) and source_attrs["whole_body_garment"].get("is_present"):
        wearing_type = "one_piece"
    else:
        wearing_type = "two_pieces"

    return {
        "sample_id": row.get("sample_id") or row.get("source_sample_id"),
        "pair_mode": row.get("pair_mode") or "paired",
        "scene_id": row.get("scene_id") or edit_task.get("scene_id"),
        "wearing_type": wearing_type,
        "person_upper_state": row.get("person_upper_state"),
        "garment_class": row.get("garment_class"),
        "person_anchor_id": row.get("person_anchor_id"),
        "garment_donor_id": row.get("garment_donor_id"),
        "template_version": instruction.get("template_version"),
        "person_image": copy_image(person_rel),
        "garment_image": copy_image(garment_rel),
        "active_region": row.get("active_region") or edit_task.get("region"),
        "edit_type_id": edit_task.get("edit_type_id"),
        "edit_type_label": edit_task.get("edit_type_label"),
        "changes": flatten_changes(changes),
        "instruction_en": instruction.get("instruction_en"),
        "instruction_zh": instruction.get("instruction_zh"),
        "instruction_spec": instruction.get("instruction_spec"),
        "preserved_by_dimension": edit_task.get("preserved_by_dimension"),
        "source_attributes": row.get("source_attributes"),
        "target_attributes": row.get("target_attributes"),
    }


def write_samples_js(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("window.SHARE_SAMPLES = ")
        json.dump(samples, handle, ensure_ascii=False, indent=2)
        handle.write(";\n")


def write_standalone_index(output_dir: Path, samples: list[dict[str, Any]]) -> None:
    template = (WEB_DIR / "viewer" / "index.html").read_text(encoding="utf-8")
    css = (WEB_DIR / "viewer" / "viewer.css").read_text(encoding="utf-8")
    js = (WEB_DIR / "viewer" / "viewer.js").read_text(encoding="utf-8")
    samples_inline = (
        "<script>\nwindow.SHARE_SAMPLES = "
        + json.dumps(samples, ensure_ascii=False)
        + ";\n</script>"
    )

    html = template
    html = html.replace(
        '  <link rel="stylesheet" href="viewer.css" />\n  <!-- BUILD:inline-styles -->',
        f"  <style>\n{css}\n  </style>",
    )
    html = html.replace("  <!-- BUILD:inline-samples -->", f"  {samples_inline}")
    html = html.replace(
        '  <script src="viewer.js"></script>\n  <!-- BUILD:inline-script -->',
        f"  <script>\n{js}\n  </script>",
    )

    (output_dir / "index.html").write_text(html, encoding="utf-8")
    # Keep separate assets for debugging/local dev.
    (output_dir / "viewer.css").write_text(css, encoding="utf-8")
    (output_dir / "viewer.js").write_text(js, encoding="utf-8")
    write_samples_js(output_dir / "data" / "samples.js", samples)


def write_extract_notice(output_dir: Path) -> None:
    text = """请先完整解压本 zip，再打开 index.html

不要从压缩包内直接双击 index.html（Windows 会显示样本列表 0，图片也无法加载）。

正确步骤：
1. 右键 zip -> 解压到文件夹
2. 进入解压后的文件夹
3. 双击 index.html

解压后目录应同时包含：
- index.html
- images/
- data_files/
"""
    (output_dir / "请先解压.txt").write_text(text, encoding="utf-8")


def copy_data_files(output_dir: Path) -> None:
    data_dir = output_dir / "data_files"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in DATA_FILES:
        src = DATA_DIR / name
        if src.is_file():
            shutil.copy2(src, data_dir / name)


def write_open_instructions(output_dir: Path) -> None:
    text = """# 分享包使用说明

1. 右键解压 `constrcut_instruction_share.zip` 到任意文件夹（必须完整解压）
2. 进入解压后的文件夹，双击打开 `index.html`
3. 若看到顶部黄色提示，说明仍未完整解压或路径不对

## 内容

- `index.html` — 单文件查看器（数据与脚本已内嵌，400 条样本）
- `images/` — 本地图片副本（必须与 index.html 同级解压）
- `data_files/` — 原始 jsonl / csv / README
- `请先解压.txt` — 打开说明

## 注意

- 不要从 zip 压缩包内直接打开 index.html
- 支持 Chrome / Edge / Firefox 本地打开
"""
    (output_dir / "OPEN.md").write_text(text, encoding="utf-8")


def create_zip(source_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(source_dir))


def main() -> int:
    args = parse_args()
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    images_dir = args.output_dir / "images"
    samples: list[dict[str, Any]] = []

    for jsonl_name in (
        "annotations_paired_with_edit.jsonl",
        "annotations_unpair_with_edit.jsonl",
    ):
        rows = read_jsonl(DATA_DIR / jsonl_name)
        for row in rows:
            if not row.get("edit_task"):
                continue
            samples.append(to_viewer_row(row, project_root=args.project_root, images_dir=images_dir))

    samples.sort(key=lambda item: (item.get("pair_mode") or "", item.get("sample_id") or ""))
    write_standalone_index(args.output_dir, samples)
    copy_data_files(args.output_dir)
    write_open_instructions(args.output_dir)
    write_extract_notice(args.output_dir)

    print(f"samples={len(samples)}")
    print(f"output_dir={args.output_dir}")

    if not args.skip_zip:
        create_zip(args.output_dir, args.zip)
        size_mb = args.zip.stat().st_size / 1024 / 1024
        print(f"zip={args.zip} ({size_mb:.1f} MB)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
