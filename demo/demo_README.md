demo部分的主要代码在该目录下。
服务器上的当前具体路径：（后期还会整理）
/data1/virtual_tryon/qwen/Qwen-image/src/examples/app.py
/data1/virtual_tryon/qwen/Qwen-image/src/examples/score_eval_demo.py
/data1/virtual_tryon/qwen/Qwen-image/src/examples/index.html


后端运行：
```bash
conda activate qwenedit
cd /data1/virtual_tryon/qwen/Qwen-Image/src/examples
PYTHONPATH=/data1/virtual_tryon/FireRed-Image-Edit:$PYTHONPATH \
CUDA_VISIBLE_DEVICES=0,1 \
python app.py --port 7860
```

要先做端口转发ssh -L 7860:localhost:7860 streetshow@222.20.96.78 -p 223

再双击打开index.html

具体功能介绍请看视频

服务器上score_eval_demo.py中需要填写apikey

错误：gpu1加载不了omnigen2模型