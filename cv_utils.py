"""
通用 CV IO 工具（中文路径兼容）
================================
原 pressure_gauge_detect.py 中与"第二步传统识别"无关的公共函数迁移至此，
供 yolo_pipeline 下标注/增强/评估/打点/推理脚本复用。
"""
from __future__ import annotations

import os

import cv2
import numpy as np


def imread_any(path: str) -> np.ndarray:
    """兼容 Windows 中文路径的 imread（返回 BGR ndarray，失败为 None）。"""
    buf = np.fromfile(path, dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_any(path: str, img: np.ndarray) -> None:
    """兼容 Windows 中文路径的 imwrite（按扩展名编码）。"""
    ext = os.path.splitext(path)[1]
    if not ext:
        ext = ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        buf.tofile(path)
