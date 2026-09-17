# -*- coding: utf-8 -*-
"""YOLO 检测框 -> 直裁图片集（框裁，割圆停用后的第一步产物）
=============================================================================
2026-09-15 起，第一步的「割圆（refine_circle + crop_circle）」在 webapp_v2 里
被**注释停用**，第二步分类器的输入改成「YOLO 检测框直接裁图」。本文件就是这条
新口径的**离线批量版**：把源图目录跑一遍第一步 YOLO，按检测框裁出定尺寸方图，
落盘成一个可直接拿去标注/训练/评估的图片集。

⚠ 与 tools/gauge_circle_crop.py 的关系：
  两者是**互斥的两条口径**，不是新旧关系。
    gauge_circle_crop.py  割圆：以盘心为中心、x/y 同倍率重采样，盘外涂黑，盘面居中
    gauge_box_crop.py     框裁：按检测框原样切，保留框内背景/其他物体
  同样的源图，两条口径产出的图片集**不可混用、不可混训**。要恢复割圆就把
  webapp_v2/server.py 里那段注释放开，并同步把这里停用 —— 口径必须与推理端一致。

⚠ 为什么 clamp_box/crop_box 定义在**本文件**而不是 webapp_v2/server.py：
  webapp_v2/server.py（在线推理）与本工具（离线出集）必须产出一模一样的裁切图，
  否则就是"训练/推理口径不一致"这类静默劣化。所以只保留这一份实现，
  webapp_v2/server.py 直接 `from gauge_box_crop import clamp_box, crop_box`。
  改这里的裁切逻辑 = 同时改变离线图片集与线上推理，两边一起变，天然同步。

输出:
  <outdir>/<stem>.jpg        框裁方图（默认 960x960）
  <outdir>/box_info.json     每张图的框/尺寸/参数记录（含未检出清单）
  <outdir>/../<name>_sheet.png  （可选 --sheet）一张联络表，肉眼核对框切得准不准

用法:
  python tools/gauge_box_crop.py -i img_temp -o labeling/box_crops_v2
      [--yolo yolo_pipeline/output/best.pt] [--conf 0.10] [--imgsz 1216]
      [--size 960] [--interp auto] [--sharpen auto] [--overlay 目录]
      [--sheet 联络表.png]
"""
from __future__ import annotations

import os
import sys
import glob
import math
import json
import argparse

import numpy as np
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "tools"))
from cv_utils import imread_any, imwrite_any            # noqa: E402
from gauge_circle_crop import (INTERP_UP, INTERP_DOWN,  # noqa: E402
                               auto_sharpen, unsharp, parse_sharpen)

DEFAULT_YOLO = os.path.join(BASE, "yolo_pipeline", "output", "best.pt")
# 检测推理分辨率：必须等于 ckpt 的训练分辨率（本 ckpt = 1216，见
# yolo_pipeline/output/best.pt 的 overrides.imgsz）。写死成显式默认值而不是 0，
# 是为了让「离线出集」和「webapp_v2 线上」用同一个数字，不去依赖 ultralytics
# 的 ckpt 回读行为。
DET_IMGSZ = 1216
CROP_SIZE = 960
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")


# ---------------------------------------------------------------- 裁切基元
def clamp_box(box, width, height):
    """把 YOLO 的浮点框夹进画面并取整 -> (x1, y1, x2, y2)。

    x1/y1 用 floor、x2/y2 用 ceil，保证裁出来的图**完全覆盖**原框，
    不会因为取整各少切掉半个像素。退化框(宽或高 < 1px)兜底成 1px，
    避免下游拿到空数组。
    """
    H, W = int(height), int(width)
    x1 = int(max(0, min(W - 1, math.floor(float(box["x1"])))))
    y1 = int(max(0, min(H - 1, math.floor(float(box["y1"])))))
    x2 = int(max(x1 + 1, min(W, math.ceil(float(box["x2"])))))
    y2 = int(max(y1 + 1, min(H, math.ceil(float(box["y2"])))))
    return x1, y1, x2, y2


def crop_box(img_bgr, box, out_size=None, interp=None, sharpen=None):
    """按第一步的检测框直接裁图（割圆停用期间的替代口径）。

    输出契约刻意与 `gauge_circle_crop.crop_circle` 保持一致，这样它俩能互换：
      out_size 非 0 → 恒返回 S×S 的 uint8 BGR 方图（第二步分类器只吃方图）
      out_size 0/None → 返回框的原生尺寸裁图（不缩放，用来肉眼核对框切得准不准）
      interp  None 自动（放大 lanczos4 / 缩小 area），或传 cv2.INTER_* 常量
      sharpen 同 crop_circle：None 自动按倍率 / False 关闭 / float / (amount, sigma)

    ⚠ 与圆裁的两点差别，改口径前必须知道：
    1) 圆裁是「以盘心为中心、x/y 同倍率」的重采样，方图几何不变形；
       框裁在 out_size 非 0 且框不是正方形时，x/y 倍率不同 → 轻微非等比拉伸。
       实测 YOLO 框接近方形（457x446 / 2042x1862 / 1440x1503，长宽比 0.96~1.10），
       拉伸量很小；要完全避免就传 out_size=0。
    2) 圆裁会把盘外像素涂黑、把圆盘摆到画面正中；框裁保留框内的一切
       （背景、其他物体），表盘在画面里的位置与占比都随框走。
       预测分布因此会和圆裁口径不同，两者结果不能直接横向比较。
    """
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = clamp_box(box, W, H)
    patch = img_bgr[y1:y2, x1:x2]

    if not out_size:
        return patch

    S = int(out_size)
    ph, pw = patch.shape[:2]
    sx, sy = S / float(pw), S / float(ph)

    # 只要有任一方向在缩小，就用面积平均 —— 抗锯齿无条件正确，
    # 所以 --interp 同样只用来挑放大方向的核（与 crop_circle 的取舍一致）。
    if min(sx, sy) < 1.0:
        interp = INTERP_DOWN
    elif interp is None:
        interp = INTERP_UP
    out = cv2.resize(patch, (S, S), interpolation=interp)

    scale = (sx + sy) / 2.0          # 锐化档位按两方向的平均倍率取
    if sharpen is None:
        amount, sigma = auto_sharpen(scale)
    elif sharpen is False:
        amount, sigma = 0.0, 0.0
    elif isinstance(sharpen, (int, float)):
        amount, sigma = float(sharpen), max(0.8, 0.35 * scale)
    else:
        amount, sigma = float(sharpen[0]), float(sharpen[1])
    if amount > 0:
        out = unsharp(out, amount, sigma)
    return out


# ---------------------------------------------------------------- 联络表
def make_sheet(tiles, path, cols=6, cell=240):
    """把 (标题, BGR图) 拼成一张联络表，用来肉眼核对 28 张框裁图。"""
    if not tiles:
        return False
    rows = int(math.ceil(len(tiles) / float(cols)))
    pad, head = 6, 22
    W = cols * cell + pad * (cols + 1)
    H = rows * (cell + head) + pad * (rows + 1)
    sheet = np.full((H, W, 3), 245, np.uint8)
    for i, (title, img) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = pad + c * (cell + pad)
        y = pad + r * (cell + head + pad)
        sheet[y:y + head, x:x + cell] = (255, 255, 255)
        cv2.putText(sheet, title, (x + 4, y + head - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1,
                    cv2.LINE_AA)
        th, tw = img.shape[:2]
        sc = min(cell / float(tw), cell / float(th))
        rs = cv2.resize(img, (max(1, int(tw * sc)), max(1, int(th * sc))),
                        interpolation=cv2.INTER_AREA)
        oy = y + head + (cell - rs.shape[0]) // 2
        ox = x + (cell - rs.shape[1]) // 2
        sheet[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    return bool(imwrite_any(path, sheet))


# ---------------------------------------------------------------- 入口
def main():
    ap = argparse.ArgumentParser(
        description="YOLO 检测框 -> 直裁图片集（框裁，割圆停用后的第一步产物）")
    ap.add_argument("-i", "--input", required=True, help="源图文件或目录")
    ap.add_argument("-o", "--outdir", required=True, help="图片集输出目录")
    ap.add_argument("--yolo", default=DEFAULT_YOLO)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--imgsz", type=int, default=DET_IMGSZ,
                    help="检测推理分辨率；必须等于 ckpt 训练分辨率，默认 1216")
    ap.add_argument("--size", type=int, default=CROP_SIZE,
                    help="输出方图边长，默认 960；0=保留检测框原生尺寸")
    ap.add_argument("--interp", default="auto",
                    choices=["auto"] + sorted(
                        ["lanczos4", "cubic", "linear", "area", "nearest"]),
                    help="上采样插值；auto=放大 lanczos4 / 缩小 area")
    ap.add_argument("--sharpen", default="auto",
                    help="USM 锐化: auto(默认, 按倍率) | off | amount | amount,sigma")
    ap.add_argument("--overlay", default=None,
                    help="可选：把检测框画在原图上的可视化目录")
    ap.add_argument("--sheet", default=None,
                    help="可选：输出一张联络表 png，肉眼核对框裁结果")
    args = ap.parse_args()

    if not os.path.exists(args.yolo):
        sys.exit(f"YOLO 权重不存在: {args.yolo}")

    if os.path.isdir(args.input):
        paths = sorted(
            p for p in glob.glob(os.path.join(args.input, "*"))
            if p.lower().endswith(IMG_EXT))
    else:
        paths = [args.input]
    if not paths:
        sys.exit(f"源图目录里没有图片: {args.input}")

    from ultralytics import YOLO
    yolo = YOLO(args.yolo)
    ck_imgsz = (dict(getattr(yolo, "overrides", {}) or {})).get("imgsz")
    if ck_imgsz and int(ck_imgsz) != int(args.imgsz):
        print(f"[warn] ckpt 训练分辨率 {ck_imgsz} != 本次推理 imgsz {args.imgsz}；"
              f"训练/推理分辨率不一致会让框位置与置信度系统性偏移")
    print(f"源图目录 : {os.path.abspath(args.input)}  ({len(paths)} 张)")
    print(f"检测权重 : {args.yolo}")
    print(f"检测分辨率: {args.imgsz}   输出: {args.size}px   "
          f"interp={args.interp}  sharpen={args.sharpen}")

    os.makedirs(args.outdir, exist_ok=True)
    if args.overlay:
        os.makedirs(args.overlay, exist_ok=True)
    interp = None if args.interp == "auto" else {
        "lanczos4": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC,
        "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
        "nearest": cv2.INTER_NEAREST}[args.interp]
    sharpen = parse_sharpen(args.sharpen)

    rows, missed, tiles = [], [], []
    print(f"\n{'image':>24s}  {'conf':>5s}  {'box(w x h)':>14s}  "
          f"{'crop':>9s}  scale")
    for p in paths:
        img = imread_any(p)
        name = os.path.splitext(os.path.basename(p))[0]
        if img is None:
            missed.append({"file": name, "src": p, "reason": "read_fail"})
            print(f"{name:>24s}  读取失败")
            continue
        H, W = img.shape[:2]
        res = yolo.predict(img, conf=args.conf, imgsz=int(args.imgsz),
                           verbose=False)
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            missed.append({"file": name, "src": p, "reason": "no_detection"})
            print(f"{name:>24s}  未检出表盘")
            continue
        b = res[0].boxes[0]
        x1f, y1f, x2f, y2f = [float(v) for v in b.xyxy[0]]
        box = {"x1": x1f, "y1": y1f, "x2": x2f, "y2": y2f,
               "cx": (x1f + x2f) / 2, "cy": (y1f + y2f) / 2,
               "w": x2f - x1f, "h": y2f - y1f, "conf": float(b.conf[0])}

        crop = crop_box(img, box, out_size=(args.size or None),
                        interp=interp, sharpen=sharpen)
        bx1, by1, bx2, by2 = clamp_box(box, W, H)
        imwrite_any(os.path.join(args.outdir, name + ".jpg"), crop)

        bw, bh = bx2 - bx1, by2 - by1
        scale = round(float(args.size) / float(min(bw, bh)), 3) if args.size else 1.0
        rows.append({
            "file": name,
            "src": p,
            "src_size": [int(W), int(H)],
            "box": {k: round(float(v), 2) for k, v in box.items()},
            "box_int": [bx1, by1, bx2, by2],
            "box_wh": [int(bw), int(bh)],
            "conf": round(float(box["conf"]), 4),
            "imgsz": int(args.imgsz),
            "crop_mode": "box",
            "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
            "crop_scale": scale,
            "clamped": bool(bx1 != math.floor(x1f) or by1 != math.floor(y1f)
                            or bx2 != min(W, math.ceil(x2f))
                            or by2 != min(H, math.ceil(y2f))),
            "aspect": round(float(bw) / float(bh), 4),
            "interp": args.interp,
            "sharpen": args.sharpen,
        })
        print(f"{name:>24s}  {box['conf']:5.2f}  "
              f"{bw:6d} x{bh:5d}  {crop.shape[1]:>4d}x{crop.shape[0]:<4d}  {scale:.2f}")

        if args.overlay:
            vis = img.copy()
            cv2.rectangle(vis, (bx1, by1), (bx2, by2), (255, 255, 0), 2)
            cv2.putText(vis, f"{name}  conf={box['conf']:.2f}", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            imwrite_any(os.path.join(args.overlay, name + ".jpg"), vis)
        if args.sheet:
            tiles.append((f"{name}  {bw}x{bh}", crop))

    info = {
        "crop_mode": "box",
        "src_dir": os.path.abspath(args.input),
        "yolo": args.yolo,
        "imgsz": int(args.imgsz),
        "conf": float(args.conf),
        "out_size": int(args.size),
        "interp": args.interp,
        "sharpen": args.sharpen,
        "n_src": len(paths),
        "n_crop": len(rows),
        "n_missed": len(missed),
        "items": rows,
        "missed": missed,
    }
    with open(os.path.join(args.outdir, "box_info.json"), "w",
              encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    if args.sheet and tiles:
        if make_sheet(tiles, args.sheet):
            print(f"\n联络表: {args.sheet}")

    print(f"\n完成: 源图 {len(paths)} 张 -> 框裁 {len(rows)} 张"
          f"{f'，未检出 {len(missed)} 张' if missed else ''}")
    print(f"图片集: {os.path.abspath(args.outdir)}")
    if missed:
        print("未检出: " + "、".join(m["file"] for m in missed))


if __name__ == "__main__":
    main()
