# annoted-code

虚拟试衣数据集标注与编辑指令构造流水线。根目录脚本负责 VLM 标注与后处理；`constrcut_instruction/` 负责场景分类、编辑类型分配与测试集导出；`match-unpair/` 负责 unpair 样本匹配。

## 目录结构

```
annoted-code/
├── annotate_api.py              # 主标注脚本（Agnes / Rivo / Ark API）
├── prompt_en.py                 # 标注 system + user prompt
├── prompt_supplement_en.py      # 补标字段 prompt
├── postprocess_annotations.py   # 标注后处理（规范化、补标、layering 重标）
├── relabel_missing_garment_annotations.py  # 补标缺失的 garment_image
├── merge_postprocessed_rows.py  # 子集后处理并合并回全量 JSONL
├── fix_sleeveless_sleeve_state.py           # 无袖区域 sleeve_state 修正
├── annotation_schema_annotated.jsonc        # JSON 字段说明（JSONC）
├── annotations_step01_base.jsonl            # 标注输入清单（sample_id + 图片路径）
├── annotations_api.jsonl                    # API 原始标注输出
├── annotations_api_postprocessed.jsonl      # 后处理后的标注
├── model_rawsay/                # VLM 原始文本输出（按 provider__model 命名）
├── match-unpair/                # unpair 样本构建
│   ├── build_unpair_by_scene.py
│   ├── sync_unpair_from_postprocessed.py
│   ├── annotations_api_unpair_200.jsonl
│   └── annotations_api_unpair_scene_manifest.jsonl
└── constrcut_instruction/       # 编辑指令流水线
    ├── pipeline/                # 核心脚本与配置
    │   ├── scene_classify.py           # 场景分类（A/B/C…）
    │   ├── label_paired_scenes.py      # paired 样本打 scene 标签
    │   ├── sync_paired_scenes_from_postprocessed.py
    │   ├── assign_and_render.py        # 分配 E01–E20 编辑类型并渲染指令
    │   ├── export_testset.py           # 导出 FireRed 测试集
    │   ├── validate_edit_pipeline.py
    │   ├── edit_common.py / edit_semantics.py
    │   ├── edit_type_catalog.json      # 编辑类型目录
    │   └── edit_value_enums.json       # 编辑字段枚举
    ├── data/                    # 流水线产物
    │   ├── annotations_paired_scenes.jsonl
    │   ├── annotations_paired_with_edit.jsonl
    │   ├── annotations_unpair_with_edit.jsonl
    │   ├── paired_edit_assignments.jsonl / unpair_edit_assignments.jsonl
    │   ├── testset_paired.jsonl / testset_unpair.jsonl
    │   └── *_edit_type_summary.csv
    └── web/                     # 标注/编辑结果浏览器
        ├── viewer/              # 开发用 viewer
        ├── build_share_bundle.py
        └── share_bundle/        # 打包后的静态分享页（可重建）
```

## 数据文件说明

| 文件 | 作用 |
|------|------|
| `annotations_step01_base.jsonl` | 待标注样本列表，每行含 `person_image` / `garment_image` 路径 |
| `annotations_api.jsonl` | VLM 标注原始 JSON 输出 |
| `annotations_api_postprocessed.jsonl` | 经 `postprocess_annotations.py` 规范化后的标注 |
| `match-unpair/annotations_api_unpair_200.jsonl` | 200 条 unpair 样本（跨人-衣配对） |
| `constrcut_instruction/data/annotations_*_with_edit.jsonl` | 带编辑类型与渲染指令的最终标注 |
| `constrcut_instruction/data/testset_*.jsonl` | 供 FireRed 评测的扁平测试集 |
| `annotation_schema_annotated.jsonc` | 全部字段语义与来源标注（模型/脚本/人工） |

## 典型流程

1. `annotate_api.py` → `annotations_api.jsonl`
2. `postprocess_annotations.py` → `annotations_api_postprocessed.jsonl`
3. `label_paired_scenes.py` → `annotations_paired_scenes.jsonl`
4. `assign_and_render.py` → `annotations_*_with_edit.jsonl`
5. `export_testset.py` → `testset_*.jsonl`

unpair 分支：`build_unpair_by_scene.py` 或 `sync_unpair_from_postprocessed.py` → 再走步骤 4–5。

## 依赖与环境

- Python 3.11+，标准库 + `urllib`（无第三方包）
- API Key 环境变量：`RIVO_API_KEY` / `AGNES_API_KEY` / `ARK_API_KEY`（按 provider 选用）
- 脚本内默认 `PROJECT_ROOT = /data1/virtual_tryon`，数据集路径为 `Datasets/eval_firsttest/`；换机器需改 `annotate_api.py` 中 `PROJECT_ROOT` 与 JSONL 内图片路径

