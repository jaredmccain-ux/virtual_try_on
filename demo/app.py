"""
虚拟试穿风格编辑 Demo — 三模型对比版（+OmniGen2）
启动: python app.py [--share] [--port 7860]

架构说明：每个模型 slot 运行在独立子进程中，进程隔离保证：
  - process.kill() 可随时中断加载或推理
  - CUDA driver 随进程退出自动回收显存
  - 两个 slot 完全独立，互不干扰
"""

import argparse
import os
import random
import sys
import threading
import time
from io import BytesIO

import gradio as gr
import multiprocessing as mp
import numpy as np
import torch
from PIL import Image

# ════════════════════════════════════════════════════════════════════
# ★ 用户需要填写/确认的区域
# ════════════════════════════════════════════════════════════════════

# FireRed pipeline 导入（需要 PYTHONPATH 指向 FireRed 目录）
from diffusers import QwenImageEditPlusPipeline
from utils.fast_pipeline import load_fast_pipeline

# Prompt 改写工具（不需要可注释掉）
from tools.prompt_utils import polish_edit_prompt

# 评估模块
from score_eval_demo import evaluate

# [TODO] 模型权重路径
MODEL_PATHS = {
    "Qwen-Image-Edit":    "/data1/virtual_tryon/qwen/pretrained/Qwen-Image-Edit-2511",
    "FireRed-Image-Edit": "/data1/virtual_tryon/FireRed-Image-Edit/pretrained/FireRed-Image-Edit-1.1",
    "OmniGen2":           "/data1/virtual_tryon/omnigen2/Models/OmniGen2",
}
# [TODO] 每个 slot 可见的 GPU（CUDA_VISIBLE_DEVICES 格式）
#   • 单卡够用时：0: "0", 1: "1"
#   • Qwen 需要双卡（>48 GB）时：0: "0,1", 1: "2"
SLOT_GPU_VISIBILITY = {
    0: "0",
    1: "1",
}

# 两个 slot 启动时默认加载的模型
SLOT_DEFAULTS = {
    0: "Qwen-Image-Edit",
    1: "FireRed-Image-Edit",
}

# ════════════════════════════════════════════════════════════════════

MAX_SEED    = np.iinfo(np.int32).max
MODEL_CHOICES = list(MODEL_PATHS.keys())

PRESETS = [
    "Make the top loose and oversized with dropped shoulders. Keep color and identity unchanged.",
    "Tuck the shirt fully into the pants. Keep everything else unchanged.",
    "Change the neckline from round neck to V-neck. Keep color and sleeve length unchanged.",
    "Extend the sleeves to long sleeves. Keep everything else unchanged.",
    "Change the top color to white. Keep garment structure and identity unchanged.",
    "Add an open black leather jacket layered over the top. Keep identity unchanged.",
]


# ════════════════════════════════════════════════════════════════════
# 子进程函数（以下所有函数在 worker 子进程内运行）
# ════════════════════════════════════════════════════════════════════

def _img_to_bytes(img: Image.Image) -> bytes:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _bytes_to_img(data: bytes) -> Image.Image:
    return Image.open(BytesIO(data)).copy()


def _load_pipeline(model_name: str):
    """
    在子进程内加载模型。
    调用前已设好 CUDA_VISIBLE_DEVICES。
    """
    dtype = torch.bfloat16

    if model_name == "Qwen-Image-Edit":
        pipe = QwenImageEditPlusPipeline.from_pretrained(
            MODEL_PATHS[model_name],
            torch_dtype=dtype,
            device_map="balanced",
        )
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        return pipe

    elif model_name == "FireRed-Image-Edit":
        pipe = load_fast_pipeline(
            MODEL_PATHS[model_name],
            device="cuda:1",
        )
        return pipe

    elif model_name == "OmniGen2":
        # OmniGen2 的导入需要 omnigen2 目录在 sys.path 中
        import sys as _sys
        _og2_path = "/data1/virtual_tryon/omnigen2"
        if _og2_path not in _sys.path:
            _sys.path.insert(0, _og2_path)

        from omnigen2.pipelines.omnigen2.pipeline_omnigen2 import OmniGen2Pipeline
        from omnigen2.models.transformers.transformer_omnigen2 import OmniGen2Transformer2DModel

        pipe = OmniGen2Pipeline.from_pretrained(
            MODEL_PATHS[model_name],
            torch_dtype=dtype,
            trust_remote_code=True,
        )
        pipe.transformer = OmniGen2Transformer2DModel.from_pretrained(
            MODEL_PATHS[model_name],
            subfolder="transformer",
            torch_dtype=dtype,
        )
        pipe = pipe.to("cuda")
        return pipe

    else:
        raise ValueError(f"Unknown model: {model_name}")


def _run_inference(pipe, model_name: str, person: Image.Image,
                   garment: Image.Image, req: dict):
    """在子进程内执行一次推理"""
    gen = torch.Generator(device="cpu").manual_seed(req["seed"])

    if model_name == "Qwen-Image-Edit":
        return pipe(
            image=[person, garment],
            prompt=req["prompt"],
            negative_prompt=" ",
            num_inference_steps=int(req["steps"]),
            true_cfg_scale=req["guidance"],
            generator=gen,
            num_images_per_prompt=int(req["num_images"]),
        ).images

    elif model_name == "FireRed-Image-Edit":
        return pipe(
            image=[person, garment],
            prompt=req["prompt"],
            negative_prompt=" ",
            num_inference_steps=int(req["steps"]),
            true_cfg_scale=req["guidance"],
            generator=gen,
            num_images_per_prompt=int(req["num_images"]),
        ).images

    elif model_name == "OmniGen2":
        w, h = person.size
        results = pipe(
            prompt=req["prompt"],
            input_images=[person, garment],
            width=w,
            height=h,
            max_input_image_side_length=2048,
            max_pixels=1024 * 1024,
            num_inference_steps=int(req["steps"]),
            max_sequence_length=1024,
            text_guidance_scale=req.get("text_guidance", 5.0),
            image_guidance_scale=req.get("image_guidance", 2.0),
            cfg_range=(req.get("cfg_start", 0.0), req.get("cfg_end", 1.0)),
            negative_prompt=req.get("negative_prompt", ""),
            num_images_per_prompt=int(req["num_images"]),
            generator=gen,
            output_type="pil",
        )
        return results.images

    else:
        raise ValueError(f"Unknown model: {model_name}")


def model_worker(model_name: str, cuda_visible: str,
                 req_q: mp.Queue, res_q: mp.Queue, status_q: mp.Queue):
    """
    子进程入口：
      1. 设置 CUDA_VISIBLE_DEVICES（必须早于任何 CUDA 操作）
      2. 加载模型，通过 status_q 上报状态
      3. 循环接收推理请求，结果写入 res_q
    进程被 kill() 后，CUDA driver 自动释放全部显存。
    """
    import signal as _signal
    _signal.signal(_signal.SIGTERM, lambda s, f: sys.exit(0))

    # 限制本进程可见的 GPU，必须在 CUDA init 之前设置
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible

    status_q.put(("status", "loading"))
    try:
        pipe = _load_pipeline(model_name)
    except Exception as e:
        status_q.put(("status", f"error: {e}"))
        return

    status_q.put(("status", "ready"))

    while True:
        req = req_q.get()
        if req is None:          # 哨兵值，正常退出
            break
        try:
            person  = _bytes_to_img(req["person"])
            garment = _bytes_to_img(req["garment"])
            images  = _run_inference(pipe, model_name, person, garment, req)
            res_q.put(("ok", [_img_to_bytes(img) for img in images]))
        except Exception as e:
            res_q.put(("error", str(e)))


# ════════════════════════════════════════════════════════════════════
# 主进程：Slot 管理
# ════════════════════════════════════════════════════════════════════

class ModelSlot:
    """管理单个 GPU slot 的子进程生命周期"""

    def __init__(self, slot_id: int, cuda_visible: str):
        self.slot_id        = slot_id
        self.cuda_visible   = cuda_visible
        self.model_name     = None
        self.process        = None
        self.req_q          = None
        self.res_q          = None
        self.status_q       = None
        self.current_status = "idle"   # 由后台 monitor 线程维护
        self._gen           = 0        # 版本号，防止旧 monitor 线程写入新状态

    @staticmethod
    def _drain(q):
        if q is None:
            return
        while True:
            try:
                q.get_nowait()
            except Exception:
                break

    def _start_monitor(self, status_q, gen: int):
        """后台线程：消费 status_q，更新 current_status"""
        def _run():
            try:
                while True:
                    kind, msg = status_q.get(timeout=300)
                    if self._gen != gen:        # 已被新 start() 替换，退出
                        return
                    if kind == "status":
                        if msg == "ready":
                            self.current_status = "ready"
                            return
                        elif msg.startswith("error"):
                            self.current_status = f"error: {msg}"
                            return
            except Exception:
                if self._gen == gen:
                    self.current_status = "timeout"

        threading.Thread(target=_run, daemon=True).start()

    def kill(self):
        """强制终止子进程；CUDA driver 自动回收其全部显存"""
        if self.process is not None and self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=10)
        self.process = None
        for q in (self.req_q, self.res_q, self.status_q):
            self._drain(q)
        self.model_name = None

    def start(self, model_name: str):
        """kill 旧进程 → 等显存释放 → 启动新 worker 进程"""
        self.kill()
        time.sleep(0.3)         # 给 CUDA driver 留时间回收显存

        self._gen          += 1
        gen                 = self._gen
        self.model_name     = model_name
        self.current_status = "loading"

        self.req_q    = mp.Queue()
        self.res_q    = mp.Queue()
        self.status_q = mp.Queue()

        self.process = mp.Process(
            target=model_worker,
            args=(model_name, self.cuda_visible,
                  self.req_q, self.res_q, self.status_q),
            daemon=True,
        )
        self.process.start()
        self._start_monitor(self.status_q, gen)

    def is_ready(self) -> bool:
        return (self.current_status == "ready"
                and self.process is not None
                and self.process.is_alive())


slots: dict = {}   # slot_id → ModelSlot，在 main() 中初始化


# ════════════════════════════════════════════════════════════════════
# Gradio 回调（运行在主进程的 Gradio handler 线程中）
# ════════════════════════════════════════════════════════════════════

def _fmt(sid: int) -> str:
    s    = slots[sid].current_status
    name = slots[sid].model_name or ""
    if s == "ready":
        return f'<span class="st-ok">■ {name} ONLINE</span>'
    if s == "loading":
        return f'<span class="st-loading">▸ {name} INITIALIZING...</span>'
    if s.startswith("error"):
        return f'<span class="st-error">✖ {s}</span>'
    if s == "timeout":
        return f'<span class="st-error">✖ LOAD TIMEOUT</span>'
    return f'<span class="st-loading">▸ {s}</span>'


def poll_initial_status():
    """页面打开时调用：阻塞直到两个 slot 完成初始加载，更新状态栏"""
    def _wait(sid):
        while True:
            s = slots[sid].current_status
            if s in ("ready", "timeout") or s.startswith("error"):
                return
            time.sleep(0.5)

    threads = [threading.Thread(target=_wait, args=(i,)) for i in range(2)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=310)

    return _fmt(0), _fmt(1)


def make_switch_fn(slot_id: int):
    """返回切换指定 slot 模型的 generator（yield HTML 字符串，实时刷新状态栏）"""
    def switch(model_name: str):
        yield f'<span class="st-loading">▸ TERMINATING Slot {slot_id}...</span>'
        slots[slot_id].start(model_name)
        yield f'<span class="st-loading">▸ {model_name} LOADING...</span>'
        while True:
            s = slots[slot_id].current_status
            if s == "ready":
                yield f'<span class="st-ok">■ {model_name} ONLINE</span>'
                break
            if s.startswith("error") or s == "timeout":
                yield f'<span class="st-error">✖ {s}</span>'
                break
            time.sleep(0.5)
    return switch


def make_generate_fn(slot_id: int):
    """返回指定 slot 的推理回调（Gradio 兼容的普通函数，非 generator）"""
    def generate_fn(person, garment, prompt_text, seed, randomize_seed,
                    steps, text_guidance, image_guidance,
                    rewrite_prompt, num_images, negative_prompt,
                    cfg_start, cfg_end):
        if person is None or garment is None:
            gr.Warning("请上传人物图和服装图")
            return [], seed
        if not prompt_text.strip():
            gr.Warning("请输入编辑指令")
            return [], seed

        if randomize_seed:
            seed = random.randint(0, MAX_SEED)

        if rewrite_prompt:
            try:
                prompt_text = polish_edit_prompt(prompt_text, person)
                print(f"[INFO] Rewritten: {prompt_text}")
            except Exception as e:
                print(f"[WARN] Rewrite failed: {e}")

        req = {
            "person":          _img_to_bytes(person),
            "garment":         _img_to_bytes(garment),
            "prompt":          prompt_text,
            "seed":            int(seed),
            "steps":           steps,
            "guidance":        text_guidance,
            "num_images":      num_images,
            "text_guidance":   text_guidance,
            "image_guidance":  image_guidance,
            "negative_prompt": negative_prompt,
            "cfg_start":       cfg_start,
            "cfg_end":         cfg_end,
        }

        slot = slots[slot_id]
        if not slot.is_ready():
            gr.Warning(f"Slot {slot_id} ({slot.model_name}) 未就绪，请等待加载完成")
            return [], seed

        slot.req_q.put(req)
        try:
            kind, data = slot.res_q.get(timeout=600)
            if kind == "ok":
                return [_bytes_to_img(b) for b in data], seed
            else:
                gr.Warning(f"Slot {slot_id} ({slot.model_name}): {data}")
                return [], seed
        except Exception as e:
            gr.Warning(f"Slot {slot_id} ({slot.model_name}): {e}")
            return [], seed

    return generate_fn


# ════════════════════════════════════════════════════════════════════
# UI + 启动
# ════════════════════════════════════════════════════════════════════

CSS = """
/* ══════════════════════════════════════════════════════════════
   ACADEMIC IFRAME THEME
   针对 GitHub Pages <iframe> 嵌入优化
   ══════════════════════════════════════════════════════════════ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Noto+Sans:wght@400;500;600;700&display=swap');

/* ── iframe 内全局 ── */
html, body {
    height: auto !important;     /* 禁止 100vh，iframe 内不能用 */
    overflow-y: auto !important;
}

/* ── 容器：iframe 模式用 100% 宽度，standalone 限宽 ── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background: #ffffff !important;
    font-family: 'Inter', 'Noto Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: #4a4a4a !important;
    border: none !important;
    box-shadow: none !important;
}
/* iframe 内隐藏 Gradio 自己的顶部工具栏和 footer */
.footer, .gradio-footer, .api-docs, .svelte-1nxz1pn { display: none !important; }

/* ── 标题区 ── */
#hero-header {
    text-align: center;
    padding: 2rem 1rem 1.5rem;
    background: #ffffff;
}
#hero-header h1 {
    font-family: 'Inter', 'Google Sans', sans-serif !important;
    font-size: 1.8rem !important;
    font-weight: 700 !important;
    color: #1a1a1a !important;
    letter-spacing: -0.5px !important;
    margin: 0 0 0.4rem !important;
    line-height: 1.25 !important;
}
#hero-header .subtitle {
    font-size: 0.95rem !important;
    color: #777 !important;
    margin: 0.4rem 0 1rem !important;
    font-weight: 400 !important;
}
.publication-links { margin-top: 0.25rem; }
.link-block {
    display: inline-block;
    margin: 0.15rem;
}
.link-block .button {
    background-color: #363636 !important;
    border: none !important;
    border-radius: 290486px !important;
    color: #fff !important;
    padding: 0.4em 1.1em !important;
    font-size: 0.82rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    text-decoration: none !important;
    display: inline-flex !important;
    align-items: center !important;
    gap: 0.3rem !important;
    transition: background-color 0.2s ease !important;
}
.link-block .button:hover {
    background-color: #2a2a2a !important;
}
.link-block .button:focus-visible {
    outline: 2px solid #3273dc !important;
    outline-offset: 2px !important;
}

/* ── 区段背景 ── */
.section-light {
    background: #f5f5f5 !important;
    padding: 1.25rem 0 !important;
    margin: 0.25rem 0 !important;
    border-radius: 6px !important;
}

/* ── 输入图 ── */
.input-img {
    border-radius: 6px !important;
    border: 1px solid #ddd !important;
    box-shadow: 0 0.25em 0.5em -0.125em rgba(10,10,10,0.08) !important;
}

/* ── 2×2 Grid ── */
.main-grid {
    display: grid !important;
    grid-template-columns: repeat(2, 1fr) !important;
    gap: 1.5rem !important;
}
.main-grid > .gr-col {
    width: auto !important;
    max-width: none !important;
    min-width: 0 !important;
    padding: 0 !important;
}

/* ── Slot Cards ── */
.slot-card {
    background: #ffffff !important;
    border-radius: 6px !important;
    box-shadow: 0 0.5em 1em -0.125em rgba(10,10,10,0.1), 0 0 0 1px rgba(10,10,10,0.02) !important;
    padding: 0.75rem 1rem !important;
    margin-bottom: 0.75rem !important;
}
.slot-card-A { border-left: 3px solid #3273dc !important; }
.slot-card-B { border-left: 3px solid #00d1b2 !important; }

/* ── Gallery ── */
.gallery {
    border-radius: 6px !important;
    border: 1px solid #ddd !important;
    overflow: hidden !important;
}

/* ── Status ── */
.st-ok { color: #00d1b2 !important; font-weight: 600 !important; }
.st-loading { color: #ffaa00 !important; font-weight: 600 !important; }
.st-error { color: #f14668 !important; font-weight: 600 !important; }

/* ── Generate Buttons (per-slot) ── */
#gen-btn-A {
    background-color: #3273dc !important;
    border: none !important;
    border-radius: 290486px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    color: #fff !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
#gen-btn-A:hover {
    background-color: #2366d1 !important;
    transform: translateY(-1px) !important;
}
#gen-btn-B {
    background-color: #00d1b2 !important;
    border: none !important;
    border-radius: 290486px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    color: #fff !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}
#gen-btn-B:hover {
    background-color: #00b89c !important;
    transform: translateY(-1px) !important;
}

/* ── Evaluate Buttons ── */
#eval-btn-A, #eval-btn-B {
    border: none !important;
    border-radius: 290486px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
    color: #fff !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    font-size: 0.85rem !important;
    min-height: 2.2rem !important;
}
#eval-btn-A {
    background-color: #6c7a89 !important;
}
#eval-btn-A:hover {
    background-color: #5a6775 !important;
    transform: translateY(-1px) !important;
}
#eval-btn-B {
    background-color: #6c7a89 !important;
}
#eval-btn-B:hover {
    background-color: #5a6775 !important;
    transform: translateY(-1px) !important;
}

/* ── Eval Result Cards ── */
.eval-card-A, .eval-card-B {
    background: #fafbfc !important;
    border-radius: 6px !important;
    padding: 0.85rem 1rem !important;
    margin-top: 0.5rem !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    border: 1px solid #e8ecef !important;
    color: #363636 !important;
}
.eval-card-A { border-left: 3px solid #3273dc !important; }
.eval-card-B { border-left: 3px solid #00d1b2 !important; }
.eval-card-A h3, .eval-card-B h3 {
    font-size: 0.92rem !important;
    font-weight: 700 !important;
    margin: 0.6rem 0 0.3rem !important;
    color: #2c3e50 !important;
}
.eval-card-A h3:first-child, .eval-card-B h3:first-child {
    margin-top: 0 !important;
}
.eval-card-A p, .eval-card-B p,
.eval-card-A li, .eval-card-B li,
.eval-card-A span, .eval-card-B span,
.eval-card-A em, .eval-card-B em,
.eval-card-A strong, .eval-card-B strong,
.eval-card-A hr + *, .eval-card-B hr + * {
    color: #363636 !important;
}
.eval-card-A ul, .eval-card-B ul {
    margin: 0.2rem 0 !important;
    padding-left: 1.2rem !important;
}
.eval-card-A li, .eval-card-B li {
    margin-bottom: 0.25rem !important;
}

/* ── Accordion ── */
.advanced-panel {
    border: 1px solid #e0e0e0 !important;
    border-radius: 6px !important;
    background: #fafafa !important;
}
.advanced-panel .label-wrap span,
.advanced-panel summary span {
    color: #363636 !important;
    font-weight: 600 !important;
}

/* ── 表单元素 ── */
.gradio-container select, .gradio-container input, .gradio-container textarea {
    border-color: #d0d0d0 !important;
    border-radius: 4px !important;
}
.gradio-container label {
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    color: #444 !important;
    font-size: 0.85rem !important;
}

/* ── Footer（iframe 内不显示，standalone 时显示）── */
#footer {
    text-align: center;
    padding: 1.5rem 1rem;
    color: #666 !important;
    font-size: 0.85rem;
    border-top: 1px solid #eee;
    font-family: 'Inter', sans-serif !important;
}
"""


def main():
    global slots

    # CUDA 不兼容 Linux 默认的 fork，必须用 spawn
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    # 初始化 slot 并后台启动默认模型
    for sid, cuda_visible in SLOT_GPU_VISIBILITY.items():
        slots[sid] = ModelSlot(sid, cuda_visible)
        slots[sid].start(SLOT_DEFAULTS[sid])
        print(f"[INFO] Slot {sid}: loading '{SLOT_DEFAULTS[sid]}' on GPU(s) {cuda_visible}")

    _gr_ver = tuple(int(x) for x in gr.__version__.split(".")[:2])

    # 使用 Soft 主题获得更干净的基底，配合 CSS 覆盖 nerfies 风格
    _theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
    )

    # Gradio 6.0: css/theme 放 launch()；4.x 放 Blocks()
    if _gr_ver >= (6, 0):
        _blocks_kwargs = {}
        _launch_kwargs = dict(css=CSS, theme=_theme)
    else:
        _blocks_kwargs = dict(css=CSS, theme=_theme)
        _launch_kwargs = {}

    with gr.Blocks(**_blocks_kwargs, title="Virtual Try-On Style Editor") as demo:
        # ═══════════════════════════════════════════════════════════
        # 输入图区域（light section）
        # ═══════════════════════════════════════════════════════════
        with gr.Column(elem_classes="section-light"):
            with gr.Row(elem_classes="main-grid"):
                # ── 左上：人物图 ──
                with gr.Column(min_width=0, elem_classes="gr-col"):
                    person = gr.Image(
                        label="人物图", type="pil",
                        height=380, elem_classes="input-img"
                    )

                # ── 右上：服装图 ──
                with gr.Column(min_width=0, elem_classes="gr-col"):
                    garment = gr.Image(
                        label="服装图", type="pil",
                        height=380, elem_classes="input-img"
                    )

        # ── 共享控制区域 ─────────────────────────────────────────
        with gr.Column(elem_classes="section-light"):
            with gr.Row():
                preset = gr.Dropdown(
                    PRESETS, value=PRESETS[0], label="预设指令",
                    allow_custom_value=True, scale=4,
                )
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="编辑指令", value=PRESETS[0], lines=2, scale=4,
                )

            preset.change(lambda x: x, inputs=[preset], outputs=[prompt_input])

            with gr.Accordion("⚙ 高级设置", open=False, elem_classes="advanced-panel"):
                with gr.Row():
                    seed           = gr.Slider(0, MAX_SEED, value=42, step=1, label="Seed")
                    randomize_seed = gr.Checkbox(label="随机 Seed", value=True)
                with gr.Row():
                    steps          = gr.Slider(1, 50,  value=50,  step=1,   label="推理步数")
                    text_guidance  = gr.Slider(1.0, 10.0, value=4.0, step=0.1, label="Text CFG")
                    image_guidance = gr.Slider(1.0, 3.0,  value=2.0, step=0.1, label="Image CFG")
                with gr.Row():
                    rewrite_prompt = gr.Checkbox(
                        label="启用 Prompt 改写（需 Qwen-VL-Max API）", value=False,
                    )
                    num_images = gr.Slider(1, 4, value=1, step=1, label="每次生成张数")
                with gr.Row():
                    negative_prompt = gr.Textbox(
                        label="Negative Prompt", lines=2,
                        value="(((deformed))), blurry, over saturation, bad anatomy, disfigured, poorly drawn face",
                    )
                with gr.Row():
                    cfg_start = gr.Slider(0.0, 1.0, value=0.0, step=0.1, label="CFG Range Start")
                    cfg_end   = gr.Slider(0.0, 1.0, value=1.0, step=0.1, label="CFG Range End")

        # ── 模型结果区域（各含独立 Generate 按钮）──────────────────
        with gr.Row(elem_classes="main-grid"):
            # ── Slot A ──
            with gr.Column(min_width=0, elem_classes="gr-col"):
                with gr.Group(elem_classes="slot-card slot-card-A"):
                    model_sel_0 = gr.Dropdown(
                        choices=MODEL_CHOICES, value=SLOT_DEFAULTS[0],
                        label="模型 A", container=False,
                    )
                    status_0 = gr.Markdown(
                        value='<span class="st-loading">▸ BOOTstrapping...</span>',
                    )
                result_0 = gr.Gallery(
                    label="结果 A", type="pil",
                    columns=2, height=480, object_fit="contain",
                    interactive=False,
                    elem_classes="gallery"
                )
                gen_btn_A = gr.Button(
                    "▶ Generate A", elem_id="gen-btn-A",
                )
                eval_btn_A = gr.Button(
                    "📋 Evaluate A", elem_id="eval-btn-A",
                )
                eval_result_A = gr.Markdown(
                    value="",
                    elem_classes="eval-card-A",
                )

            # ── Slot B ──
            with gr.Column(min_width=0, elem_classes="gr-col"):
                with gr.Group(elem_classes="slot-card slot-card-B"):
                    model_sel_1 = gr.Dropdown(
                        choices=MODEL_CHOICES, value=SLOT_DEFAULTS[1],
                        label="模型 B", container=False,
                    )
                    status_1 = gr.Markdown(
                        value='<span class="st-loading">▸ BOOTstrapping...</span>',
                    )
                result_1 = gr.Gallery(
                    label="结果 B", type="pil",
                    columns=2, height=480, object_fit="contain",
                    interactive=False,
                    elem_classes="gallery"
                )
                gen_btn_B = gr.Button(
                    "▶ Generate B", elem_id="gen-btn-B",
                )
                eval_btn_B = gr.Button(
                    "📋 Evaluate B", elem_id="eval-btn-B",
                )
                eval_result_B = gr.Markdown(
                    value="",
                    elem_classes="eval-card-B",
                )

        # ── 事件绑定 ──────────────────────────────────────────────
        demo.load(fn=poll_initial_status, inputs=None, outputs=[status_0, status_1])

        model_sel_0.change(fn=make_switch_fn(0), inputs=[model_sel_0], outputs=[status_0])
        model_sel_1.change(fn=make_switch_fn(1), inputs=[model_sel_1], outputs=[status_1])

        common_inputs = [
            person, garment, prompt_input, seed, randomize_seed,
            steps, text_guidance, image_guidance,
            rewrite_prompt, num_images, negative_prompt,
            cfg_start, cfg_end,
        ]

        gen_btn_A.click(
            fn=make_generate_fn(0),
            inputs=common_inputs,
            outputs=[result_0, seed],
        )
        gen_btn_B.click(
            fn=make_generate_fn(1),
            inputs=common_inputs,
            outputs=[result_1, seed],
        )
        prompt_input.submit(
            fn=make_generate_fn(0),
            inputs=common_inputs,
            outputs=[result_0, seed],
        )
        prompt_input.submit(
            fn=make_generate_fn(1),
            inputs=common_inputs,
            outputs=[result_1, seed],
        )

        eval_btn_A.click(
            fn=evaluate,
            inputs=[person, garment, result_0, prompt_input],
            outputs=[eval_result_A],
        )
        eval_btn_B.click(
            fn=evaluate,
            inputs=[person, garment, result_1, prompt_input],
            outputs=[eval_result_B],
        )

        # ── 点击 Gallery 图片 → 新标签页打开原图 ──────────────────
        gr.HTML("""
        <script>
        document.addEventListener('click', function(e) {
            var img = e.target.closest('img');
            if (!img) return;
            var gallery = img.closest('.gallery, [data-testid="gallery"], .gradio-gallery');
            if (!gallery) return;
            e.preventDefault();
            e.stopPropagation();
            window.open(img.src, '_blank');
        }, true);
        </script>
        """, visible=False)

    demo.queue().launch(server_name="0.0.0.0", server_port=args.port, share=args.share,
                        **_launch_kwargs)


if __name__ == "__main__":
    main()
