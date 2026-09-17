# -*- coding: utf-8 -*-
"""灭火器压力表识别 —— 网页界面 v2（YOLO-pose 6 关键点版 · 双模型）
=========================================================================
流水线:

  ① YOLO 检出表盘 bbox              —— classifier/classify_image_yolo.py
  ② 按检测框直裁（默认 960×960）     —— tools/gauge_box_crop.py
  ③ **椭圆 -> 正圆校正**              —— trans/dial_ellipse_to_circle.py
       把裁切图里的椭圆表盘白化校正成正圆（圆心落在图像正中心，圆外填充
       (30,30,30)）。失败或判定不可信时**自动降级**用②的直裁图，不丢结果。
  ④ YOLO-pose 定位**6 个关键点**     —— classifier/classify_pose.py
       双模型（当前口径）:
         geom 模型（5 点）: kpt0 表盘圆心 / kpt1 欠压边界(红区起点)
                            / kpt2 欠压·正常分界 / kpt3 正常·超压分界
                            / kpt4 超压边界(黄区终点)
         tip  模型（1 点）: 指针尖端
       合并回 6 点 —— kpt0 圆心 ... kpt5 针尖（与旧 6 点单模型逐槽对齐）
       单模型（向后兼容）: kpt_shape=[1,3] / [2,3] / [5,3] / [6,3] 都能跑
  ⑤ 在**校正图**上把 6 个点画出来（带点号）
       -> 回传 pose_jpeg，网页里直接当图片显示

⚠ 为什么中间要插一层正圆校正（2026-09-17）
  侧视/斜拍时表盘在图像上是**椭圆**。圆心 -> 针尖的连线与各条边界连线的夹角，
  被透视**非等比压缩**过 —— 同一个真实角度，沿长轴量与沿短轴量结果不一样大。
  压力三态判据本质就是比角度，所以不校正，角度本身就是错的。
  白化校正把椭圆映成正圆后，角度才回到真实比例。
  ⚠ 坐标要反算：模型是在校正图上出点的，必须用 Minv 映射回裁切图、再映射回
    原图（见 predict 里的 _to_patch）。少了这层，页面上的点会整体错位。
  ⚠ 校正有失败可能，且**信任是有代价的**：实测 10__原图 曾因 Hough 兜底把圆心
    定错（偏离图心 0.23·短边），表盘被推出画面，针尖误差从 2px 飙到 215px。
    因此 correct_to_circle 里加了「圆心必须靠近裁切图中心」的校验，
    不可信就退回直裁图 —— 宁可少校正几张，也不交一张错的。
  A/B 对比脚本：tools/check_rectify_ab.py（同一张图跑两条口径，比点误差与三态）

⚠ 为什么要拆成两个模型（2026-09-16 晚）
  单个 6 点模型里，关键点 loss 是 6 个点**求平均**，针尖偏 300px 与偏 3px
  对总 loss 的贡献都只有 1/6，梯度被 5 个静止点淹没 -> 模型放弃针尖，
  输出与表盘绑定的固定角（实测 r≈0.049）。拆开后两侧 loss 各自 100%
  作用在自己的点上，针尖不再被稀释。
  拆分的实现：classifier/train_pose.py --task tip|geom
              tools/split_pose6_into_two.py（数据派生）

⚠ 2026-09-17：**本版判压力状态（欠压/正常/超压）**。口径恢复为「6 点 + 状态」
  两步都出。判据（classifier/classify_pose.py 的 state_from_kpts）：
  把圆心到 5 个非圆心点的射线都画出来，看「圆心 -> 针尖」这条被夹在哪两条
  相邻的边界射线之间 —— 落在 欠压边界~欠压/正常分界 = 欠压，依此类推。
  判不了时（边界点置信度不足 / 4 个边界点角度次序被模型串了 / 点数不够）
  返回 state=None + state_reason，页面显示「未判定」，**不拿固定角度区间兜底**。

  2026-09-16 晚曾经收窄为「只展示 6 个点、不判状态」，原因是当时边界点精度
  不够，现恢复状态输出。批量实测（121 张，脚本在主工程 tools/
  report_pressure_state.py）：用人工关键点判定 97.5%（判据本身可靠），
  用模型预测点 88.0% —— 差的全是针尖模型在个别源上跑飞
  （点误差 525~862px，几乎整个表盘尺度）。
  ⚠ 别拿「指针长度比」（|圆心->针尖| / |圆心->边界|）当在线自检：实测判对组
    中位 1.21、判错组 1.22，两组分布**完全重叠** —— 跑飞是"指错方向"，
    不是"指得特别远"。能分开好坏的只有"点误差"，而它推理时拿不到，
    所以**目前没有可用的在线自检**。

  架构名**不写死**：启动时从 ckpt 的 train_args 读出来（classify_pose.
  arch_name），经 /api/health 回显到页面，换 n/s/l/x 权重不用改任何代码。

  割圆（refine_circle + crop_circle）延续 2026-09-15 的结论保持停用；
  第二步的输入 = **检测框直裁图再经椭圆校正**（见上）。
  ⚠ 因此它与离线数据集 crops_box* 不再逐字节同源 —— 数据集里没有校正这一层。
    这是有意为之（校正是给线上补几何畸变），但**要拿这些图重训 pose 时得
    把校正一起做进去**，否则又是训练/推理口径不一致。

  另有两套基于「针尖周围颜色」的独立判定实现（不参与本流水线，留作对照）：
    classifier/tip_zone.py    单圆 25px + 四色计数
    classifier/tip_state.py   法线双圆 5px + 半圆兜底

只依赖 Python 标准库 http.server 起服务，不引入 flask / fastapi。

启动（默认 127.0.0.1:8770，不自动开浏览器；Windows 直接双击 webapp_v2/start.bat）:
  # 本包默认即双模型，权重路径已按包内目录结构配好，直接跑就行
  E:\\Anaconda\\envs\\wanhua\\python.exe webapp_v2/server.py
  E:\\Anaconda\\envs\\wanhua\\python.exe webapp_v2/server.py --port 8770 --open
  # 换成自训权重时显式指定
  E:\\Anaconda\\envs\\wanhua\\python.exe webapp_v2/server.py \\
      --weights-tip <针尖权重.pt> --weights-geom <几何权重.pt>
  # 单模型（向后兼容，kpt_shape 支持 [1,3] / [2,3] / [5,3] / [6,3]）
  E:\\Anaconda\\envs\\wanhua\\python.exe webapp_v2/server.py --weights <单模型.pt>
  # 局域网/服务器对外访问时自行指定: --host 0.0.0.0
  # CPU 上嫌慢: --no-pose-jpeg

接口:
  GET  /              页面 (index.html)
  GET  /api/health    健康检查 + 当前参数（含两个模型各自的 arch/params/kpt_shape）
  GET  /api/config    页面初始化用的默认参数
  POST /api/predict   单张图片字节流(Content-Type: image/*) -> JSON
                      查询参数(都可省略, 缺省即用服务端默认):
                        name=文件名        (URL 编码, 仅用于回显)
                        conf=0.10          第一步 YOLO 检测置信度阈值
                        kpt_conf=0.20      第二步关键点置信度阈值
                        det_imgsz          —— 已锁定 1216（检测器训练分辨率）
                        pose_imgsz         —— 已锁定 960（pose 训练分辨率）
                                              两者传了也不会生效
                        crop_size=960      裁切输出边长; 0=保留框的原生尺寸
                        interp=auto        auto|lanczos4|cubic|linear|area|nearest
                        sharpen=auto       auto|off|0.7|0.7,1.5
                        rectify=1          1=启用椭圆->正圆校正(默认)
                                           0=跳过校正，直裁图直接喂第二步(A/B 用)
                        want_crop=1        1=回传**校正后**的正圆图(base64 jpeg)
                        want_pose=1        1=回传画了 6 个关键点的校正图(base64 jpeg)
                        hough_max_side=800 割圆停用期间只被接收、不使用
"""
from __future__ import annotations

import os
import sys
import json
import time
import base64
import argparse
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs, unquote

import numpy as np
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # 包根
sys.path.insert(0, BASE)
sys.path.insert(0, os.path.join(BASE, "classifier"))
sys.path.insert(0, os.path.join(BASE, "yolo_pipeline"))
sys.path.insert(0, os.path.join(BASE, "tools"))
sys.path.insert(0, os.path.join(BASE, "trans"))

# 第一步：表盘检测（复用生产推理入口的函数与默认权重路径）
import classify_image_yolo as CY  # noqa: E402

# 裁切图 -> 椭圆校正为正圆（接在第一步与第二步之间）
import dial_ellipse_to_circle as RECT  # noqa: E402

# 第二步：YOLO-pose 指针尖定位（针尖 = 最后一个关键点）
import classify_pose as CP  # noqa: E402

from gauge_circle_crop import HOUGH_MAX_SIDE  # noqa: E402
# clamp_box / crop_box 只保留 tools/gauge_box_crop.py 里那一份实现：
# 离线批量出图片集（gauge_box_crop.py CLI）与在线推理（本文件）必须裁出
# 逐字节相同的图，否则就是"训练/推理口径不一致"的静默劣化。
from gauge_box_crop import clamp_box, crop_box  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
INDEX = os.path.join(HERE, "index.html")

MAX_BYTES = 40 * 1024 * 1024        # 单张图片上限 40MB
JPEG_Q = 82                         # 回传裁切图的 jpeg 质量
POSE_JPEG_Q = 88                    # 关键点图质量高一点, 点要看得清

# 检测推理分辨率：锁定为训练时的 1216，不做成前端可调项。
# 本 ckpt 是 imgsz=1216 训出来的，推理分辨率必须与训练一致，
# 否则同一个表盘的框位置与置信度会系统性偏移（训练/推理不一致）。
DET_IMGSZ_FIXED = 1216

# 第二步 pose 推理分辨率：同样锁定为它的训练分辨率 960。
# ultralytics 推理默认 640，比训练分辨率低一档 —— 关键点比 bbox 对分辨率更
# 敏感（针尖只有几个像素宽），实测 640 下误差明显变大、还会整张丢检测。
POSE_IMGSZ_FIXED = CP.DEFAULT_IMGSZ

# 第二步的裁切口径：box = 按第一步检测框直裁（割圆停用）
CROP_MODE = "box"

# 第二步 pose 权重。三套路径都给上，实际用哪套由 CP.resolve_default_weights()
# 决定（单一事实来源在 classify_pose.py，本文件不重复判断逻辑）：
#   双模型（当前口径）: classifier/yolo-v8n-pose-best.pt（针尖 1 点，yolov8n-pose）
#                      + classifier/best_geom.pt      （几何 5 点，yolov8m-pose）
#   单模型（向后兼容）: classifier/best.pt（旧 6 点合一）
# 架构名一律从 ckpt 里读（arch_name），页面上显示的「yolov8n-pose」是实读的，
# 换权重不用改本文件。
DEFAULT_POSE = os.path.join(BASE, "classifier", "best.pt")
DEFAULT_POSE_TIP = os.path.join(BASE, "classifier", "yolo-v8n-pose-best.pt")
DEFAULT_POSE_GEOM = os.path.join(BASE, "classifier", "best_geom.pt")

# 前端下拉框的取值清单（与 UI 保持一致, 集中在这里便于核对）
INTERPS = ["auto"] + sorted(CY.INTERP_CHOICES)
SHARPENS = ["auto", "off"]

# status 口径：这里回答的是「针尖找没找到」（定位成没成），**不是压力状态**。
# 压力状态单独走 out["pressure"]（= classify_pose 的 state），两者别混：
# 定位失败时压力状态自然也是未判定，但定位成功也可能判不了状态。
ST_FOUND = "找到针尖"
ST_NO_TIP = "未找到针尖"
ST_NO_DIAL = "未检出表盘"


# ------------------------------------------------------------------ 工具
def decode_image(buf: bytes):
    """把上传的字节流解码成 BGR 图；失败返回 None。

    用 imdecode 而不是 imwrite+imread，既省一次落盘，也天然避开中文路径坑。
    """
    arr = np.frombuffer(buf, dtype=np.uint8)
    if arr.size == 0:
        return None
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def encode_jpeg_b64(bgr, quality=JPEG_Q) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def draw_pose_overlay(bgr, tip, conf=None, kpts=None):
    """在 BGR 图上画**全部关键点**，返回新图。

    tip 为 (x, y) 裁切图坐标；None 表示没找到针尖（原样返回）。
    kpts 是 classify_pose 回的 [{name,x,y,conf}, ...]，给全 6 点时按点号画：
        0 圆心(品红) / 1~4 色区边界(青) / 5 针尖(黄十字, 主视觉)
    只给 1~2 个点时退回旧行为（只画针尖）。

    为什么 6 个点全画：本版网页的产品就是「这 6 个点」，页面右侧同时列出
    每个点的点名/坐标/置信度，图和表对着看才能判断是哪个点偏了、偏到哪。

    标签一律用 ASCII 数字 —— cv2.putText 画不了中文（会变成 ???），中文点名交给
    网页的 HTML 文本显示（见 index.html 的关键点列表）。

    所有半径/线宽按图像边长比例算，这样 crop_size=0（原生尺寸）时观感一致。
    """
    img = bgr.copy()
    H, W = img.shape[:2]
    r = max(8, int(round(min(H, W) / 78.0)))
    L = r * 2                       # 十字臂长
    lw = max(2, int(round(min(H, W) / 260.0)))
    fs = max(0.55, min(H, W) / 1050.0)

    def _cross(x, y, color, ring=True):
        x = int(clamp(x, 0, W - 1))
        y = int(clamp(y, 0, H - 1))
        # 四臂十字：先深色描边再上色，保证任何底色上都看得清
        for w, c in ((lw + 4, (0, 0, 0)), (lw + 1, color)):
            cv2.line(img, (x - L, y), (x + L, y), c, w, cv2.LINE_AA)
            cv2.line(img, (x, y - L), (x, y + L), c, w, cv2.LINE_AA)
        if ring:
            cv2.circle(img, (x, y), r, (0, 0, 0), lw, cv2.LINE_AA)
            cv2.circle(img, (x, y), r, color, 1, cv2.LINE_AA)
            cv2.circle(img, (x, y), max(1, lw - 1), color, -1, cv2.LINE_AA)
        return x, y

    def _text(txt, x, y, color, anchor_right=False):
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, fs, lw)
        tx = x - tw - 8 if anchor_right else x
        tx = int(clamp(tx, 4, max(4, W - tw - 6)))
        ty = int(clamp(y, th + 8, max(th + 8, H - 6)))
        cv2.rectangle(img, (tx - 6, ty - th - 7), (tx + tw + 6, ty + 6),
                      (0, 0, 0), -1)
        cv2.putText(img, txt, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, color,
                    lw, cv2.LINE_AA)

    klist = list(kpts or [])
    # 1~4 号边界点先画（编号越小越靠里，避免针尖/圆心盖住它们）
    # 注：conf==0 的点是双模型合并时的占位（该模型本次没出点），跳过不画 ——
    #     否则会在 (0,0) 堆一串假的十字叉，看起来像"检出一堆垃圾点"。
    for i in range(1, len(klist) - 1):
        kp = klist[i]
        if not kp.get("conf"):
            continue
        x, y = _cross(kp["x"], kp["y"], (255, 190, 0), ring=False)
        _text(str(i), x + r + 10, y - r - 8, (255, 190, 0))
    if len(klist) > 2 and klist[0].get("conf"):     # 圆心（第 0 个）
        kp = klist[0]
        x, y = _cross(kp["x"], kp["y"], (255, 0, 255), ring=False)
        _text("0", x + r + 10, y - r - 8, (255, 0, 255))

    if tip is None or not conf:
        return img
    x, y = _cross(tip[0], tip[1], (0, 215, 255))
    parts = ["tip"]
    if conf is not None:
        parts.append("%.2f" % conf)
    parts.append("(%d,%d)" % (x, y))
    _text(" ".join(parts), x + r + 10, y - r - 8, (0, 215, 255))
    return img


# ------------------------------------------------------------------ 推理引擎
class Engine:
    """检测 + pose 模型加载一次、常驻；推理加锁串行，避免 CPU 上多请求互相拖慢。

    pose 侧支持两种模式（由传入的路径决定，构造时会打印用的是哪种）：
      双模型  pose_tip_path + pose_geom_path 都有效
              -> 各跑一次，合并回 6 点（kpt0 圆心 ... kpt5 针尖）
      单模型  pose_path（旧 6 点 best.pt）
              -> 按它自己的 kpt_shape 出点，行为与改版前一致
    """

    def __init__(self, yolo_path, pose_path, conf, kpt_conf, crop_size,
                 interp, sharpen, imgsz, want_crop=True, want_pose=True,
                 interp_name="auto", sharpen_name="auto",
                 hough_max_side=HOUGH_MAX_SIDE,
                 pose_imgsz=POSE_IMGSZ_FIXED,
                 pose_tip_path=None, pose_geom_path=None,
                 rectify=True):
        self.yolo_path = yolo_path
        self.pose_path = pose_path            # 单模型权重（可为 None）
        self.pose_tip_path = pose_tip_path    # 针尖模型（1 点）
        self.pose_geom_path = pose_geom_path  # 几何模型（5 点）
        self.conf = conf
        self.kpt_conf = kpt_conf
        self.crop_size = crop_size
        self.interp = interp
        self.sharpen = sharpen
        self.imgsz = imgsz
        self.want_crop = want_crop
        self.want_pose = want_pose
        self.pose_imgsz = int(pose_imgsz)
        # 椭圆 -> 正圆校正（trans/dial_ellipse_to_circle.py）：裁切后先校正再送 pose。
        # 关掉它就退回「检测框直裁图直接喂 pose」的老口径，便于 A/B 对比。
        self.rectify = bool(rectify)
        # 回显给页面用的原始字符串（interp/sharpen 的实参是 None/tuple，
        # 没法直接展示，分开存）
        self.interp_name = interp_name
        self.sharpen_name = sharpen_name
        self.hough_max_side = hough_max_side
        self.yolo = None
        self.pose = None            # 单模型
        self.pose_tip = None        # 针尖模型
        self.pose_geom = None       # 几何模型
        self.mode = None            # "dual" / "single"，load() 里定
        self.kpt_shape = []          # 合并后点数的形状，例 [6,3]
        self.kpt_names = []          # 例 ["表盘圆心", ..., "指针尖端"]
        # 各模型自己的信息（给 /api/health 与页面标题栏）
        self.tip_info = None         # {label, arch, params_m, kpt_shape, n_kpt}
        self.geom_info = None
        self.pose_info_one = None    # 单模型时的那一份
        self.pose_arch = None        # 单模型时的架构名（兼容旧的 health 字段）
        self.pose_params = None
        self.lock = threading.Lock()
        self.n_done = 0
        self.total_ms = 0.0

    # ---- 加载 ----
    def load(self):
        if not os.path.exists(self.yolo_path):
            raise SystemExit(f"[webapp_v2] YOLO 检测权重不存在: {self.yolo_path}")

        dual = bool(self.pose_tip_path) and bool(self.pose_geom_path)
        if not dual and not self.pose_path:
            raise SystemExit(
                "[webapp_v2] 没有任何 pose 权重。期望其中之一:\n"
                f"  双模型: {DEFAULT_POSE_TIP} + {DEFAULT_POSE_GEOM}\n"
                f"  单模型: {DEFAULT_POSE}")

        t0 = time.perf_counter()
        print(f"[webapp_v2] 加载 YOLO 检测模型: {self.yolo_path}", flush=True)
        self.yolo = CY.load_yolo(self.yolo_path)

        if dual:
            for tag, p in (("针尖模型", self.pose_tip_path),
                           ("几何模型", self.pose_geom_path)):
                if not os.path.exists(p):
                    raise SystemExit(f"[webapp_v2] {tag}不存在: {p}")
            self.mode = "dual"
            print(f"[webapp_v2] 加载 pose 针尖模型: {self.pose_tip_path}",
                  flush=True)
            self.pose_tip = CP.load_model(self.pose_tip_path)
            print(f"[webapp_v2] 加载 pose 几何模型: {self.pose_geom_path}",
                  flush=True)
            self.pose_geom = CP.load_model(self.pose_geom_path)

            self.tip_info = CP.model_info(self.pose_tip, "针尖模型")
            self.geom_info = CP.model_info(self.pose_geom, "几何模型")
            for info in (self.tip_info, self.geom_info):
                print(f"[webapp_v2]   {info['label']}  arch={info['arch']}  "
                      f"params={info['params_m']}M  kpt_shape={info['kpt_shape']}",
                      flush=True)
            # 合并后固定 6 点（geom 5 + tip 1），命名与旧 6 点单模型逐槽对齐
            n_total = int(self.geom_info["n_kpt"]) + int(self.tip_info["n_kpt"])
            self.kpt_shape = [n_total, 3]
            self.kpt_names = CP.kpt_names(n_total)
            # 兼容旧字段 pose_arch / pose_params：历史实现是「取两个模型里参数量
            # 较大的那个当代表」。⚠ 双模型下这**不能当标题用** —— 两个模型架构可以
            # 不同（当前就是针尖 yolov8n-pose + 几何 yolov8m-pose），只报一个会让
            # 人以为换的权重没生效（真踩过：针尖已换成 n 版，标题栏仍印 yolov8m）。
            # 页面已改为从 pose_models 各自实读的 arch 合成（index.html: archText），
            # 这两个字段只留给老消费者，别再加新的展示依赖。
            _rep = max((self.tip_info, self.geom_info),
                       key=lambda d: d["params_m"] or 0)
            self.pose_arch = _rep["arch"]
            self.pose_params = _rep["params_m"]
            print(f"[webapp_v2]   合并 -> kpt_shape {self.kpt_shape}  "
                  f"kpt_names {self.kpt_names}  "
                  f"（取针尖 = 最后一个关键点）", flush=True)
        else:
            if not os.path.exists(self.pose_path):
                raise SystemExit(
                    f"[webapp_v2] pose 权重不存在: {self.pose_path}")
            self.mode = "single"
            print(f"[webapp_v2] 加载 pose 模型(单模型): {self.pose_path}",
                  flush=True)
            self.pose = CP.load_model(self.pose_path)
            self.pose_info_one = CP.model_info(self.pose, "pose")
            ks = self.pose_info_one["kpt_shape"]
            self.kpt_shape = ks
            self.kpt_names = CP.kpt_names(
                int(ks[0]) if ks else 1)
            self.pose_arch = self.pose_info_one["arch"]
            self.pose_params = self.pose_info_one["params_m"]
            print(f"[webapp_v2]   架构 {self.pose_arch}  "
                  f"参数量 {self.pose_params}M  "
                  f"kpt_shape {ks or '?'}  "
                  f"nc {getattr(self.pose.model, 'nc', '?')}  "
                  f"names {getattr(self.pose.model, 'names', '?')}  "
                  f"-> 取针尖 = 最后一个关键点"
                  f"（共 {len(self.kpt_names)} 个: {self.kpt_names}）", flush=True)

        print(f"[webapp_v2]   pose 推理分辨率 {self.pose_imgsz}（= 训练分辨率）",
              flush=True)
        print(f"[webapp_v2]   椭圆->正圆校正: "
              f"{'启用' if self.rectify else '停用'}"
              f"（trans/dial_ellipse_to_circle.py；裁切后先校正再送 pose）",
              flush=True)
        print(f"[webapp_v2] 模型就绪, 用时 {time.perf_counter() - t0:.1f}s", flush=True)

    # ---- 给页面/接口用的模型信息 ----
    def pose_models_info(self):
        """返回给 /api/health 与 /api/config 的模型描述（双模型时两个都在）。"""
        def _one(info):
            if not info:
                return None
            return {"label": info["label"], "arch": info["arch"],
                    "params_m": info["params_m"],
                    "kpt_shape": info["kpt_shape"], "n_kpt": info["n_kpt"]}

        if self.mode == "dual":
            return {"mode": "dual",
                    "tip": _one(self.tip_info),
                    "geom": _one(self.geom_info),
                    # 合并口径：页面按这个顺序列 6 行点名
                    "merged_kpt_names": self.kpt_names,
                    "merged_n_kpt": len(self.kpt_names)}
        info = self.pose_info_one or {}
        return {"mode": "single",
                "single": _one(info),
                "merged_kpt_names": self.kpt_names,
                "merged_n_kpt": len(self.kpt_names)}

    # ---- 单张推理 ----
    def predict(self, img_bgr, name="", **over):
        """返回可直接 json 化的 dict。over 可覆盖本次请求的参数。"""
        conf = float(over.get("conf", self.conf))
        kpt_conf = float(over.get("kpt_conf", self.kpt_conf))
        crop_size = int(over.get("crop_size", self.crop_size))
        interp = over.get("interp", self.interp)
        sharpen = over.get("sharpen", self.sharpen)
        imgsz = int(self.imgsz)        # 固定 1216：不接受请求级覆盖
        want_crop = bool(over.get("want_crop", self.want_crop))
        want_pose = bool(over.get("want_pose", self.want_pose))
        # 允许按请求关掉校正（A/B 对比用）；默认取服务级开关
        rectify = bool(over.get("rectify", self.rectify))

        h, w = img_bgr.shape[:2]
        out = {"ok": True, "name": name, "width": int(w), "height": int(h),
               "imgsz": imgsz, "pose_imgsz": int(self.pose_imgsz),
               "bbox": None, "conf": None, "circle": None,
               "crop_box": None, "crop_size": None,
               "status": ST_NO_DIAL, "pose": None,
               "rectify": None,
               "crop_jpeg": None, "pose_jpeg": None,
               "error": None, "elapsed_ms": 0}

        t0 = time.perf_counter()
        with self.lock:
            box = CY.detect_gauge(self.yolo, img_bgr, conf, imgsz=imgsz)
            out["bbox"] = box
            out["conf"] = None if box is None else round(float(box["conf"]), 4)
            if box is None:
                out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
                self.n_done += 1
                self.total_ms += out["elapsed_ms"]
                return out

            # 割圆（refine_circle + crop_circle）延续停用：
            # 现用口径 = 直接按第一步检测框裁图，不做定圆、不做圆裁。
            patch = crop_box(img_bgr, box, out_size=(crop_size or None),
                             interp=interp, sharpen=sharpen)
            bx1, by1, bx2, by2 = clamp_box(box, w, h)

            # ---- ③ 椭圆 -> 正圆校正（trans/dial_ellipse_to_circle.py）----
            # 裁切图里的表盘在侧视/斜拍时是**椭圆**，圆心->针尖的夹角会被非等比
            # 压缩（同一真实角度，长轴方向与短轴方向量出来不一样大）。三态判据
            # 本质就是比角度，所以不校正就会偏。
            # 校正图圆心固定在正中心、圆外填充 (30,30,30)。
            # ⚠ 失败时**原样降级**用 patch 跑 —— 校正只是增强，不能因为它在
            #   某张图上没找到表盘就把这张的结果丢掉（第一步已经找到表盘了）。
            rect = None
            feed = patch                 # feed = 真正送进 pose 的那张图
            if rectify:
                rect = RECT.correct_to_circle(patch)
                if rect["ok"]:
                    feed = rect["circle"]

            # 第二步：pose 定位关键点，坐标在 **feed（校正图）** 系下。
            # 双模型时 model_tip/model_geom 都传；单模型时只传 model。
            # imgsz 必须显式给（= 该权重的训练分辨率），否则 ultralytics 按
            # 默认 640 跑，针尖精度会掉、还可能整张丢检测。
            # want_state=True：6 点齐全就顺带判压力三态（欠压/正常/超压），
            # 判据 = 圆心->针尖 的射线落在哪两条相邻边界射线之间。
            r = CP.classify(feed, model=self.pose,
                            model_tip=self.pose_tip,
                            model_geom=self.pose_geom,
                            conf_thres=kpt_conf,
                            imgsz=self.pose_imgsz, want_state=True)

        ph, pw = patch.shape[:2]
        out["crop_box"] = {"mode": "box", "x1": bx1, "y1": by1,
                           "x2": bx2, "y2": by2,
                           "w": int(bx2 - bx1), "h": int(by2 - by1),
                           "cx": round((bx1 + bx2) / 2.0, 2),
                           "cy": round((by1 + by2) / 2.0, 2)}
        out["crop_size"] = [int(pw), int(ph)]

        # feed(校正图) -> patch(裁切图) 的反算。**这一步不能省**：
        # 校正把圆心搬到了图像正中心，两者坐标原点不是同一个，不反算点会整体错位。
        # 没做校正（关闭 / 校正失败降级）时是恒等映射。
        _Minv = rect["Minv"] if (rect and rect["ok"]) else None

        def _to_patch(p):
            if _Minv is None:
                return float(p[0]), float(p[1])
            v = RECT.map_points(_Minv, np.asarray([p], dtype=np.float32))[0]
            return float(v[0]), float(v[1])

        # 只有针尖是必需的；圆心是 geom 模型（或旧 [2,3]/[6,3] 权重）白送的
        # 附带回显，不参与判定
        tip = None if r["tip"] is None else _to_patch(r["tip"])
        cen = None if r["cx"] is None else _to_patch((r["cx"], r["cy"]))
        # 另留一份 **feed 系** 的针尖：overlay 画在校正图上，坐标必须用这套，
        # 直接拿 patch 系的 tip 会画偏（校正图原点 ≠ 裁切图原点）。
        tip_feed = (None if r["tip"] is None
                    else (float(r["tip"][0]), float(r["tip"][1])))

        # 裁切图坐标 -> 原图坐标。crop_box 只做「裁剪 + 缩放到 out_size」，
        # 所以是纯仿射：src = box_left_top + crop_pt * (box_wh / crop_wh)。
        # bw/bh 与 pw/ph 不相等时两轴缩放系数不同（各向异性），必须分开算，
        # 否则画回原图会整体偏移。
        tips = cens = None
        if tip is not None:
            tips = (bx1 + tip[0] * (bx2 - bx1) / pw,
                    by1 + tip[1] * (by2 - by1) / ph)
        if cen is not None:
            cens = (bx1 + cen[0] * (bx2 - bx1) / pw,
                    by1 + cen[1] * (by2 - by1) / ph)

        # 全部关键点同样换算回原图坐标：网页要在原图上把 6 个点都画出来，
        # 只有针尖一个是没法核对「边界点有没有串」的。
        def _to_src(p):
            return (round(bx1 + p[0] * (bx2 - bx1) / pw, 2),
                    round(by1 + p[1] * (by2 - by1) / ph, 2))

        # 模型给的点在 feed 系；先整体反算回裁切图系（页面/CSV 里的「裁切图坐标」
        # 语义一直是裁切图系，不能因为内部多了一层校正就变），再换算到原图。
        kpts_patch = []
        kpts_src = []
        for kp in (r["kpts"] or []):
            px, py = _to_patch((kp["x"], kp["y"]))
            kpts_patch.append({"name": kp["name"], "x": round(px, 2),
                               "y": round(py, 2), "conf": kp["conf"]})
            sx, sy = _to_src((px, py))
            kpts_src.append({"name": kp["name"], "x": sx, "y": sy,
                             "conf": kp["conf"]})

        out["status"] = ST_FOUND if r["ok"] else ST_NO_TIP
        # 压力三态。state 为 None 时看 state_reason 知道为什么判不了
        # （置信度不足 / 边界点次序乱 / 点数不够）—— 不要拿 status 去猜。
        out["pressure"] = r.get("state")
        out["pressure_reason"] = r.get("state_reason")
        out["pose"] = {
            "n_kpt": r["n_kpt"],
            # 双模型时分别回显各模型的点数与是否出点，页面据此标红缺失的那组
            "split": r.get("split"),
            "n_kpt_tip": r.get("n_kpt_tip"),
            "n_kpt_geom": r.get("n_kpt_geom"),
            "tip": None if tip is None else {
                "x": round(tip[0], 2), "y": round(tip[1], 2),
                "conf": (None if r["tip_conf"] is None
                         else round(float(r["tip_conf"]), 4))},
            "tip_src": None if tips is None else {"x": round(tips[0], 2),
                                                  "y": round(tips[1], 2)},
            "center": None if cen is None else {"x": round(cen[0], 2),
                                                "y": round(cen[1], 2)},
            "center_src": None if cens is None else {"x": round(cens[0], 2),
                                                     "y": round(cens[1], 2)},
            "kpts": kpts_patch,          # 裁切图坐标, 带点名与置信度
            "kpts_src": kpts_src,        # 原图坐标, 画在缩略图上用这套
            "angle": None if r["angle"] is None else round(float(r["angle"]), 2),
            "coord": "crop",      # 主坐标（另给 *_src 为原图坐标）
            "ok": bool(r["ok"]),
            "reason": r["reason"],
            # ---- 压力状态（判据 = 圆心->针尖 射线被哪两条边界射线夹住）----
            # 页面画射线用：span.lo / span.hi 是 kpts / kpts_src 的**下标**，
            # 圆心固定是下标 0，针尖固定是最后一个（6 点时 = 5）。
            "state": r.get("state"),
            "state_reason": r.get("state_reason"),
            "state_span": r.get("state_span"),
            "state_margin_deg": r.get("state_margin_deg"),
            "boundary_angles": r.get("boundary_angles"),
        }
        # 椭圆校正实况。页面据此标明这张是「已校正成正圆」还是「原图直喂」，
        # 以及失败原因 —— 否则用户没法解释为什么这张的点看着和别的不一样。
        # ratio = 椭圆短轴/长轴（1.0 ≈ 本来就是正圆）；circle 模式无 ratio。
        _rok = bool(rect and rect["ok"])
        out["rectify"] = {
            "enabled": bool(rectify),
            "applied": _rok,
            "mode": (("ellipse" if rect["corrected"] else "circle")
                     if _rok else None),
            "ratio": (None if not _rok or rect["ratio"] is None
                      else round(float(rect["ratio"]), 3)),
            "r": (None if not _rok else round(float(rect["r"]), 1)),
            # 检测到的圆心偏离裁切图中心的比例（可信度指标，>0.15 会被判不可信）
            "d_center": (None if rect is None or rect.get("d_center") is None
                         else round(float(rect["d_center"]), 3)),
            "ms": (None if rect is None else int(round(rect["ms"]))),
            "reason": (None if (rect is None or _rok) else rect["reason"]),
        }
        if want_crop:
            # 回传的「裁切图」= **校正后的正圆图**（= 模型真正吃进去的那张）。
            # 不再回传未校正的检测框直裁图：页面上那张图的作用就是让用户核对
            # 「模型看到的画面」，给一张模型没看过的原裁切图只会造成误判。
            out["crop_jpeg"] = encode_jpeg_b64(feed, JPEG_Q)
        if want_pose:
            # 画在**校正后**的图（= 模型真正吃进去的那张）上：
            # 点和模型看到的画面在同一坐标系里，才看得出点准不准。
            out["pose_jpeg"] = encode_jpeg_b64(
                draw_pose_overlay(feed, tip_feed, r["tip_conf"], r["kpts"]),
                POSE_JPEG_Q)

        out["elapsed_ms"] = int((time.perf_counter() - t0) * 1000)
        self.n_done += 1
        self.total_ms += out["elapsed_ms"]
        return out

    def health(self):
        return {"ok": True,
                "mode": self.mode,
                "yolo": os.path.basename(self.yolo_path),
                "pose": os.path.basename(self.pose_path or ""),
                "pose_tip": os.path.basename(self.pose_tip_path or ""),
                "pose_geom": os.path.basename(self.pose_geom_path or ""),
                "pose_arch": self.pose_arch,
                "pose_params": self.pose_params,
                "pose_imgsz": int(self.pose_imgsz),
                "rectify": bool(self.rectify),
                "pose_models": self.pose_models_info(),
                "pose_info": {"task": "pose",
                              "arch": self.pose_arch,
                              "params_m": self.pose_params,
                              "imgsz": int(self.pose_imgsz),
                              "kpt_shape": self.kpt_shape,
                              "kpt_names": self.kpt_names,
                              "nc": (getattr(self.pose_geom or self.pose,
                                             "model", None) is not None
                                     and getattr((self.pose_geom or self.pose).model,
                                                 "nc", None)),
                              "names": (getattr(self.pose_geom or self.pose,
                                                "model", None) is not None
                                        and getattr((self.pose_geom or self.pose).model,
                                                    "names", None))},
                "n_done": self.n_done,
                "avg_ms": int(self.total_ms / self.n_done) if self.n_done else 0,
                "defaults": self.config()}

    def config(self):
        return {"conf": self.conf, "kpt_conf": self.kpt_conf, "imgsz": self.imgsz,
                "pose_imgsz": int(self.pose_imgsz),
                "crop_size": self.crop_size, "interp": self.interp_name,
                "sharpen": self.sharpen_name,
                "want_crop": self.want_crop, "want_pose": self.want_pose,
                "rectify": bool(self.rectify),
                "hough_max_side": self.hough_max_side, "crop_mode": CROP_MODE,
                "kpt_names": self.kpt_names,
                "pose_arch": self.pose_arch,
                "pose_params": self.pose_params,
                "mode": self.mode,
                "pose_models": self.pose_models_info(),
                "pose_info": {"task": "pose",
                              "arch": self.pose_arch,
                              "params_m": self.pose_params,
                              "imgsz": int(self.pose_imgsz),
                              "kpt_shape": self.kpt_shape,
                              "kpt_names": self.kpt_names},
                "statuses": [ST_FOUND, ST_NO_TIP, ST_NO_DIAL],
                "interps": INTERPS, "sharpens": SHARPENS}


# ------------------------------------------------------------------ HTTP
class Handler(BaseHTTPRequestHandler):
    server_version = "gauge-webapp-v2/2.0-pose"
    engine: Engine = None            # main() 里注入

    # ---- 输出辅助 ----
    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def log_message(self, fmt, *args):
        pass                          # 自己打，格式更紧凑

    def _log(self, code, extra=""):
        print(f'[http] {self.address_string()} "{self.command} {self.path}" '
              f'{code}{extra}', flush=True)

    # ---- GET ----
    def do_GET(self):
        path = urlparse(self.path).path
        if path in ("/", "/index.html"):
            try:
                with open(INDEX, "rb") as fh:
                    body = fh.read()
            except OSError as e:
                self._send(500, f"index.html 读取失败: {e}".encode("utf-8"),
                           "text/plain; charset=utf-8")
                self._log(500)
                return
            self._send(200, body, "text/html; charset=utf-8")
            self._log(200)
        elif path == "/api/health":
            self._json(200, self.engine.health())
            self._log(200)
        elif path == "/api/config":
            self._json(200, self.engine.config())
            self._log(200)
        elif path == "/favicon.ico":
            self._send(204, b"", "image/x-icon")
        else:
            self._json(404, {"ok": False, "error": f"未知路径 {path}"})
            self._log(404)

    # ---- POST ----
    def do_POST(self):
        u = urlparse(self.path)
        if u.path != "/api/predict":
            self._json(404, {"ok": False, "error": f"未知路径 {u.path}"})
            self._log(404)
            return

        q = parse_qs(u.query)
        one = lambda k, d=None: (q.get(k) or [d])[0]          # noqa: E731
        name = unquote(one("name", "") or "")[:180]

        try:
            over = {}
            if one("conf") not in (None, ""):
                over["conf"] = clamp(float(one("conf")), 0.01, 1.0)
            if one("kpt_conf") not in (None, ""):
                over["kpt_conf"] = clamp(float(one("kpt_conf")), 0.01, 1.0)
            # imgsz 故意不解析：检测分辨率固定为训练分辨率 1216（DET_IMGSZ_FIXED），
            # 前端传什么都不生效，避免出现"训练/推理分辨率不一致"的静默劣化。
            if one("crop_size") not in (None, ""):
                cs = int(one("crop_size"))
                over["crop_size"] = 0 if cs == 0 else int(clamp(cs, 128, 2048))
            it = one("interp")
            if it:
                if it not in INTERPS:
                    raise ValueError(f"interp 非法: {it}")
                over["interp"] = None if it == "auto" else CY.INTERP_CHOICES[it]
            sp = one("sharpen")
            if sp:
                over["sharpen"] = CY.parse_sharpen(sp)
            for key in ("want_crop", "want_pose", "rectify"):
                v = one(key)
                if v is not None and v != "":
                    over[key] = v not in ("0", "false", "False")
            hm = one("hough_max_side")
            if hm is not None and hm != "":
                over["hough_max_side"] = max(0, int(hm))
        except (TypeError, ValueError) as e:
            self._json(400, {"ok": False, "name": name, "error": f"参数错误: {e}"})
            self._log(400, f" 参数错误: {e}")
            return

        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            n = 0
        if n <= 0:
            self._json(400, {"ok": False, "name": name, "error": "空请求体"})
            self._log(400, " 空请求体")
            return
        if n > MAX_BYTES:
            self._json(413, {"ok": False, "name": name,
                             "error": f"图片超过 {MAX_BYTES // 1024 // 1024}MB"})
            self._log(413, " 过大")
            return

        buf = self.rfile.read(n)
        t0 = time.perf_counter()
        img = decode_image(buf)
        if img is None:
            self._json(400, {"ok": False, "name": name,
                             "error": "图片解码失败（格式不支持或文件损坏）"})
            self._log(400, " 解码失败")
            return

        try:
            r = self.engine.predict(img, name=name or "upload", **over)
        except Exception as e:                                # noqa: BLE001
            import traceback
            traceback.print_exc()
            self._json(500, {"ok": False, "name": name,
                             "error": f"推理异常: {type(e).__name__}: {e}"})
            self._log(500, " 推理异常")
            return

        self._json(200, r)
        self._log(200, f"  {r['status']}  {int((time.perf_counter()-t0)*1000)}ms")


# ------------------------------------------------------------------ 入口
def main():
    ap = argparse.ArgumentParser(description="灭火器压力表识别 网页界面 v2 (pose)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="监听地址。局域网/服务器对外访问改成 0.0.0.0")
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--yolo", default=CY.DEFAULT_YOLO, help="第一步检测权重")
    ap.add_argument("--weights", default=None,
                    help=f"第二步 pose **单模型**权重（旧 6 点合一），默认 "
                         f"{DEFAULT_POSE}。给了 --weights-tip/--weights-geom 时"
                         f"本项被忽略")
    ap.add_argument("--weights-tip", default=None,
                    help=f"第二步 pose **针尖模型**权重（1 点），默认 "
                         f"{DEFAULT_POSE_TIP}")
    ap.add_argument("--weights-geom", default=None,
                    help=f"第二步 pose **几何模型**权重（5 点：圆心+4边界），"
                         f"默认 {DEFAULT_POSE_GEOM}")
    ap.add_argument("--single", action="store_true",
                    help="强制用单模型（忽略针尖/几何两个专用权重，"
                         "走 --weights 或 best.pt）")
    ap.add_argument("--conf", type=float, default=0.10,
                    help="第一步 YOLO 检测置信度阈值")
    ap.add_argument("--kpt-conf", type=float, default=0.20,
                    help="第二步关键点置信度阈值，低于它算没检出")
    ap.add_argument("--imgsz", type=int, default=DET_IMGSZ_FIXED,
                    help="检测推理分辨率，默认锁定 1216（与训练一致）。"
                         "只在换用其它分辨率训练的 ckpt 时才需要改；"
                         "页面上的参数面板不提供这一项")
    ap.add_argument("--pose-imgsz", type=int, default=POSE_IMGSZ_FIXED,
                    help=f"第二步 pose 推理分辨率，默认 {POSE_IMGSZ_FIXED}"
                         "（= pose 权重的训练分辨率）。换用别的分辨率训出的"
                         "权重时才需要改")
    ap.add_argument("--crop-size", type=int, default=CY.CROP_SIZE,
                    help="裁切输出边长（0=保留检测框的原生尺寸）")
    ap.add_argument("--interp", default=CY.CROP_INTERP, choices=INTERPS)
    ap.add_argument("--sharpen", default=CY.CROP_SHARPEN)
    ap.add_argument("--hough-max-side", type=int, default=HOUGH_MAX_SIDE,
                    help="圆形精化的 Hough 降采样上限。⚠ 割圆停用期间只被接收、不使用")
    ap.add_argument("--no-pose-jpeg", dest="want_pose", action="store_false",
                    default=True, help="不回传关键点图（省带宽，页面就看不到点）")
    ap.add_argument("--no-crop-jpeg", dest="want_crop", action="store_false",
                    default=True, help="不回传原始裁切图")
    ap.add_argument("--no-rectify", dest="rectify", action="store_false",
                    default=True,
                    help="关掉「椭圆 -> 正圆」校正，直接把检测框裁切图喂第二步。"
                         "留着是为了 A/B 对比；默认开")
    ap.add_argument("--open", action="store_true", help="启动后自动打开浏览器")
    args = ap.parse_args()

    if not os.path.exists(INDEX):
        raise SystemExit(f"[webapp_v2] 缺少页面文件: {INDEX}")

    interp = None if args.interp == "auto" else CY.INTERP_CHOICES[args.interp]

    # ---- 决定用哪套 pose 权重 ----
    # 优先级：显式 --weights-* > --single/--weights > 自动探测（classify_pose
    # 的 resolve_default_weights，本文件不重复那套判断逻辑）。
    pose_path = pose_tip = pose_geom = None
    if args.weights_tip or args.weights_geom:
        pose_tip = args.weights_tip or DEFAULT_POSE_TIP
        pose_geom = args.weights_geom or DEFAULT_POSE_GEOM
    elif args.single or args.weights:
        pose_path = args.weights or DEFAULT_POSE
    else:
        _t, _g, _s = CP.resolve_default_weights()
        if _t and _g:
            pose_tip, pose_geom = str(_t), str(_g)
        elif _s:
            pose_path = str(_s)
        elif _t:
            pose_tip = str(_t)
        elif _g:
            pose_geom = str(_g)
        else:
            raise SystemExit(
                "[webapp_v2] 找不到任何 pose 权重。期望其中之一:\n"
                f"  双模型: {DEFAULT_POSE_TIP} + {DEFAULT_POSE_GEOM}\n"
                f"  单模型: {DEFAULT_POSE}\n"
                f"  先跑训练: python classifier/train_pose.py --task both")

    eng = Engine(args.yolo, pose_path, clamp(args.conf, 0.01, 1.0),
                 clamp(args.kpt_conf, 0.01, 1.0),
                 args.crop_size, interp, CY.parse_sharpen(args.sharpen),
                 max(0, args.imgsz), want_crop=bool(args.want_crop),
                 want_pose=bool(args.want_pose),
                 interp_name=args.interp, sharpen_name=args.sharpen,
                 hough_max_side=max(0, args.hough_max_side),
                 pose_imgsz=max(0, args.pose_imgsz),
                 pose_tip_path=pose_tip, pose_geom_path=pose_geom,
                 rectify=bool(args.rectify))
    eng.load()

    Handler.engine = eng
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    url = f"http://{'127.0.0.1' if args.host in ('0.0.0.0', '') else args.host}:{args.port}/"
    print(f"[webapp_v2] 已启动: {url}", flush=True)
    if eng.mode == "dual":
        print(f"[webapp_v2] 第二步 双模型: "
              f"tip={os.path.basename(eng.pose_tip_path)} "
              f"({eng.tip_info['arch']}, {eng.tip_info['params_m']}M, "
              f"kpt_shape={eng.tip_info['kpt_shape']})  +  "
              f"geom={os.path.basename(eng.pose_geom_path)} "
              f"({eng.geom_info['arch']}, {eng.geom_info['params_m']}M, "
              f"kpt_shape={eng.geom_info['kpt_shape']})", flush=True)
    else:
        print(f"[webapp_v2] 第二步 单模型: "
              f"{os.path.basename(eng.pose_path)}  "
              f"({eng.pose_arch}, {eng.pose_params}M, "
              f"kpt_shape={eng.kpt_shape})", flush=True)
    print(f"[webapp_v2]   合并 {len(eng.kpt_names)} 点 {eng.kpt_names}  "
          f"imgsz={eng.pose_imgsz}  kpt_conf={args.kpt_conf}", flush=True)
    print(f"[webapp_v2] 裁切 {args.crop_size}px  interp={args.interp}  "
          f"sharpen={args.sharpen}  det_conf={args.conf}  det_imgsz={args.imgsz}",
          flush=True)
    print("[webapp_v2] 裁切口径: 检测框直裁（割圆已注释停用）", flush=True)
    print("[webapp_v2] 输出口径: 6 个关键点(圆心+4 边界+针尖) + 压力状态"
          "(欠压/正常/超压，判据=圆心->针尖射线被夹在哪两条边界射线之间)",
          flush=True)
    print("[webapp_v2] Ctrl+C 停止", flush=True)
    if args.open:
        import webbrowser
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n[webapp_v2] 已停止", flush=True)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
