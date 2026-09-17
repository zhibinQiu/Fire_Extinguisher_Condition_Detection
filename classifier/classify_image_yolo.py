# -*- coding: utf-8 -*-
"""
灭火器压力表 端到端识别 (YOLOv8-cls 版)
========================================
流水线: YOLO 检测表盘 bbox -> 圆形精化裁切(黑底方图) -> YOLOv8n-cls 三分类
状态:   欠压 / 正常 / 超压   (4 方向旋转投票, 与 LOO 评估口径一致)

替代 classify_image.py(ResNet18) 的 YOLOv8-cls 版本。
模型: classifier/output/yolocls17/final_yolov8n_cls_gauge.pt (全量 17 张训练)
      注意该权重是用「圆形裁切」图训练的, 所以本文件的裁切方式必须用
      crop_circle; 而 classify_image.py 的权重是在「矩形 bbox 裁切」
      (labeling/crops) 上训练的 —— 两者裁切口径不同, 不可混用。

关键尺寸(2026-09-14):
  --imgsz     0    (沿用 checkpoint 的训练分辨率; 显式传值才覆盖)
  --crop-size 960  (圆形精化裁切输出边长; 目标半径 = 960/2 - 4 = 476)
  分类器输入仍是 224, 由 ultralytics 内部从 960 方形图重采样, 不要改动。
  圆裁目标半径满足 `R = size/2 - 4`, 因此 tools/gauge_3stage.py 的
  `R = w/2 - 4` 在 960 下依然精确成立, 三段式校验可直接复用该裁切图。

用法:
  python classify_image_yolo.py -i ../img/10.png [--debug]
  python classify_image_yolo.py -i ../img -o out.json [--conf 0.10]
  python classify_image_yolo.py -i ../img_temp --crop-size 960
"""
import os
import sys
import json
import glob
import argparse

import numpy as np
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "yolo_pipeline"))
sys.path.insert(0, os.path.join(BASE, "tools"))
from cv_utils import imread_any, imwrite_any  # noqa: E402
from gauge_circle_crop import (refine_circle, crop_circle,  # noqa: E402
                               INTERP_CHOICES, parse_sharpen)

CLASSES = ["欠压", "正常", "超压"]
EN_NAME = {"欠压": "qianya", "正常": "zhengchang", "超压": "chaoya"}
CN_NAME = {v: k for k, v in EN_NAME.items()}
IMGSZ = 224        # 第二步分类器输入(不要动, 与训练口径绑定)

# 第一步 YOLO 检测分辨率: 0 = 沿用 checkpoint 自带的训练分辨率。
# ultralytics 会把 ckpt 的 train args 合进 model.overrides, 所以不显式指定时
# 训推天然一致; train.py 改成 960 重训后, 这里不用动就自动跟着变。
# (实测: 用 480 权重硬跑 960 推理, 表盘命中反而从 1/9 掉到 0/9)
DET_IMGSZ = 0
# 圆形精化裁切输出边长: 960。目标半径 = 960/2 - 4 = 476,
# 与 tools/gauge_3stage.py 里写死的 `R = w/2 - 4` 精确一致。
CROP_SIZE = 960
# 圆裁的上采样质量(2026-09-14 起默认开启, 实测依据见 tools/compare_upscale.py):
#   表盘小到 r=57~130px 时, 拉到 960 是 4~8 倍上采样, 插值会把指针/刻度抹软。
#   默认 lanczos4 插值 + 按倍率自动 USM(≈0.7/1.5), 比旧的 cubic 无锐化利落
#   一档且不引入可见光晕。想复现旧观感: --interp cubic --sharpen off。
# ⚠️ labeling/circle_crops 是旧配方生成的。若要用新配方训第二步分类器,
#    必须用同一组参数重新生成那批裁切图, 否则训推不一致。
CROP_INTERP = "auto"
CROP_SHARPEN = "auto"

DEFAULT_YOLO = os.path.join(BASE, "yolo_pipeline", "output", "best.pt")
# ⚠ 旧的「第二步分类器」（纯分类，classes=欠压/正常/超压）—— 本精简包**不包含**
#   它：现在的第二步是 YOLO-pose 出关键点再算状态（classifier/classify_pose.py），
#   网页端从不加载这个 .pt。只有本文件的 CLI（--cls）会用到，届时需自行提供权重。
DEFAULT_CLS = os.path.join(BASE, "classifier", "output", "yolocls17",
                           "final_yolov8n_cls_gauge.pt")


def load_yolo(path):
    from ultralytics import YOLO
    return YOLO(path)


def detect_gauge(yolo, img_bgr, conf_thres=0.10, imgsz=DET_IMGSZ):
    """返回置信度最高 gauge bbox dict 或 None。imgsz=0 表示沿用 ckpt 训练分辨率"""
    kw = {"imgsz": int(imgsz)} if imgsz else {}
    res = yolo.predict(img_bgr, conf=conf_thres, verbose=False, **kw)
    if not res or res[0].boxes is None or len(res[0].boxes) == 0:
        return None
    b = res[0].boxes[0]
    x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
            "w": x2 - x1, "h": y2 - y1, "conf": float(b.conf[0])}


def classify_rotate(model, bgr, imgsz=IMGSZ):
    """0/90/180/270 旋转投票, 返回 (中文状态, {类:概率})"""
    names = list(model.names.values())   # e.g. ['chaoya','qianya','zhengchang']
    probs = []
    for k in range(4):
        im = bgr if k == 0 else np.rot90(bgr, k=-k)
        r = model.predict(im, imgsz=imgsz, verbose=False)[0]
        probs.append(r.probs.data.cpu().numpy())
    avg = np.mean(probs, axis=0)
    idx = int(np.argmax(avg))
    pred_en = names[idx] if idx < len(names) else "?"
    status = CN_NAME.get(pred_en, pred_en)
    prob = {c: float(avg[names.index(EN_NAME[c])]) if EN_NAME[c] in names else 0.0
            for c in CLASSES}
    return status, {k: round(v, 3) for k, v in prob.items()}


def process_one(yolo, model, img_path, conf_thres, debug_dir=None,
                crop_size=CROP_SIZE, det_imgsz=DET_IMGSZ,
                interp=None, sharpen=None):
    img_bgr = imread_any(img_path)
    if img_bgr is None:
        return {"image_path": img_path, "error": "read_fail"}
    name = os.path.splitext(os.path.basename(img_path))[0]
    box = detect_gauge(yolo, img_bgr, conf_thres, imgsz=det_imgsz)
    if box is None:
        return {"image_path": img_path, "bbox": None, "conf": None,
                "status": "未检出表盘", "prob": None}
    # 圆形精化裁切(与训练 circle_crops 同一函数); 输出固定 crop_size 见方,
    # 小表盘在此被上采样到 476 半径, 默认带 lanczos4 + 自动 USM
    cx, cy, r, method = refine_circle(img_bgr, box)
    circle_bgr = crop_circle(img_bgr, cx, cy, r,
                             out_size=(crop_size or None),
                             interp=interp, sharpen=sharpen)
    status, prob = classify_rotate(model, circle_bgr)
    circle_r = float(r)
    out = {"image_path": img_path, "bbox": box, "conf": box["conf"],
           "circle": {"cx": float(cx), "cy": float(cy), "r": circle_r,
                      "method": method},
           "crop_size": [int(circle_bgr.shape[1]), int(circle_bgr.shape[0])],
           "status": status, "prob": prob}
    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        vis = img_bgr.copy()
        x1, y1, x2, y2 = [int(v) for v in (box["x1"], box["y1"],
                                           box["x2"], box["y2"])]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.circle(vis, (int(cx), int(cy)), int(r), (255, 0, 255), 2)
        label = f"{status} {prob[status]:.2f}"
        cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 0, 0), 3)
        cv2.putText(vis, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 1)
        dbg = os.path.join(debug_dir, name + ".yolocls.jpg")
        imwrite_any(dbg, vis)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="图片文件或目录")
    ap.add_argument("-o", "--output", default=None, help="结果 JSON 路径")
    ap.add_argument("--yolo", default=DEFAULT_YOLO)
    ap.add_argument("--weights", default=DEFAULT_CLS)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=DET_IMGSZ,
                    help="第一步 YOLO 检测分辨率; 0=沿用 checkpoint 训练分辨率")
    ap.add_argument("--crop-size", type=int, default=CROP_SIZE,
                    help="圆形精化裁切输出边长, 默认 960; 0=原生尺寸")
    ap.add_argument("--interp", default=CROP_INTERP,
                    choices=["auto", "lanczos4", "cubic", "linear", "area",
                             "nearest"],
                    help="圆裁上采样插值; auto=放大 lanczos4/缩小 area")
    ap.add_argument("--sharpen", default=CROP_SHARPEN,
                    help="圆裁 USM 锐化: auto(默认, 按倍率) | off | amount | "
                         "amount,sigma")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(args.weights):
        sys.exit(f"分类器权重不存在: {args.weights}\n"
                 f"请先运行 classifier/train_yolo_cls.py --mode full")
    if not os.path.exists(args.yolo):
        sys.exit(f"YOLO 权重不存在: {args.yolo}")

    print("加载 YOLO:", args.yolo)
    yolo = load_yolo(args.yolo)
    print("加载分类器:", args.weights)
    model = load_yolo(args.weights)

    if os.path.isdir(args.input):
        paths = sorted(
            p for p in glob.glob(os.path.join(args.input, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
            and ".yolocls." not in p.lower()
            and ".yolodbg." not in p.lower())
    else:
        paths = [args.input]

    debug_dir = os.path.join(BASE, "labeling", "debug_yolocls") if args.debug else None
    crop_interp = None if args.interp == "auto" else INTERP_CHOICES[args.interp]
    crop_sharpen = parse_sharpen(args.sharpen)
    print(f"圆裁: {args.crop_size}px  interp={args.interp}  "
          f"sharpen={args.sharpen}")
    results = []
    for p in paths:
        r = process_one(yolo, model, p, args.conf, debug_dir,
                        crop_size=args.crop_size, det_imgsz=args.imgsz,
                        interp=crop_interp, sharpen=crop_sharpen)
        results.append(r)
        print(f"{os.path.basename(p):>18s}  ->  {r.get('status')}"
              + (f"  {r['prob'][r['status']]:.2f}" if r.get("prob") else ""))

    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("\n结果已写:", args.output)


if __name__ == "__main__":
    main()
