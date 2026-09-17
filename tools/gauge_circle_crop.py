# -*- coding: utf-8 -*-
"""
YOLO 矩形框 → 精确表盘圆 (圆形裁切工具)
====================================================
背景: YOLO gauge bbox 是带 pad 的外接矩形(pad≈0.25r, box≈2.5r),
含灭火器外壳/背景, 干扰下游分类。本工具把矩形精化为圆形表盘。

四种定位方法, 圆心/半径分开取:
  1) 几何先验: r0 = min(w,h)/2.5, 中心=bbox中心 (YOLO 标注的物理标定)
  2) HoughCircles: 在 bbox 内, 圆心限制在中心邻域, 半径限 [0.72,1.25]*r0
  3) 白盘面连通域: "高亮+低饱和" 盘面 mask → 最大连通域 minEnclosingCircle
  4) 射线法: 整幅原图上沿 180 条射线找"盘面(亮)->盘外(暗)"的外沿半径
圆心取 2)/3) 中与先验中心最贴合者(只有它们有真实圆心估计); 半径取 4) 并与先验
r0 取上包络。为什么不用 2)/3) 的半径见 refine_circle 的注释。任一失败自动降级,
本函数永不失败。

⚠️ 2026-09-15 修正 "割圆割得不对":
  旧版把半径也交给 2)/3) 融合, 但两者的半径都是**单边偏小**的 ——
  hough 在 bbox 子图上跑, maxr = 1.25*r0 = min(w,h)/2, 恰好是框短边的一半,
  检测框一旦贴紧表盘(短边≈2r)真实半径就顶出搜索窗之外;
  white_dial 的掩膜是 V>120 & S<90, 天然把彩色色带排除, minEnclosingCircle
  只圈得住白色盘心。半径偏小会让 crop_scale = 476/r 偏大, 盘面外圈色带被推出
  画幅直接切掉(不可逆)。实测 19 张有人工真值的图: r管线/r真值 均值 0.890,
  ±10% 内仅 9/19, 最差 0.63。改用射线法后均值 1.050, 偏小>15% 的张数 3→0。

输出:
  - 圆形裁切图 (默认 960×960 方块, 圆外可填充黑/白)
  - 也可叠加在原图上输出可视化

输出尺寸 (--size):
  默认 960。目标半径 = size/2 - pad, 因此 tools/gauge_3stage.py 里写死的
  `R = w/2 - 4` 在 960 下依然精确成立(476), 三段式下游无需改动。
  --size 0 退回原生尺寸(边长 2r+2pad), 即旧行为。

上采样质量 (--interp / --sharpen):
  表盘在原图里常常只有 r=57~130px, 拉到固定 960(目标半径 476)就是 4~8 倍
  上采样, 插值会把指针细线和刻度抹软。默认组合是实测调出来的:
    --interp auto   放大用 INTER_LANCZOS4(边缘比 CUBIC 利落), 缩小用 INTER_AREA
    --sharpen auto  按放大倍率做圆内 USM 锐化, 4~8 倍区间约 amount 0.7 / sigma 1.5
  依据见 tools/compare_upscale.py(逐档扫插值 × amount × sigma, 量化锐度/越界
  能量/过曝, 并输出肉眼对照网格)。想复现旧观感: --interp cubic --sharpen off。
  ⚠️ 改了 --interp/--sharpen 就等于改了第二步分类器的输入口径。labeling/
     circle_crops 是旧配方生成的, 若要用新的圆裁图训分类器, 需用同一组参数
     重新生成, 否则训推不一致。

推理分辨率 (--imgsz):
  默认 0 = 沿用 checkpoint 自带的训练分辨率。ultralytics 会把 ckpt 的 train args
  合进 model.overrides, 所以只要不显式指定, 训推分辨率天然一致,
  retrain 之后无需改这里的代码。
  ⚠️ 实测结论(2026-09-14, 旧 480 权重 + 9 张实景图):
     把推理硬改成 960(权重还是 480 训的), 有框率 8/9 → 9/9,
     但精化圆命中真表盘 1/9 → 0/9, 反而更差。
     分辨率改大不能替代重新训练 —— 必须先用 train.py --imgsz 960 重训。

用法:
  python gauge_circle_crop.py -i ../img_temp -o ../labeling/circle_crops
      [--yolo ../yolo_pipeline/output/best.pt] [--fill black]
      [--overlay ../labeling/circle_overlay] [--conf 0.10] [--scale 1.25]
      [--size 960] [--imgsz 0] [--interp auto] [--sharpen auto]
"""
import os
import sys
import glob
import math
import argparse

import numpy as np
import cv2

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 包根
sys.path.insert(0, BASE)
from cv_utils import imread_any, imwrite_any  # noqa: E402

DEFAULT_YOLO = os.path.join(BASE, "yolo_pipeline", "output", "best.pt")


# ---------------- 三种圆定位 ----------------

def circle_prior(box: dict):
    """① 几何先验: YOLO 标注 box≈2.5r, 中心≈圆心"""
    cx = box["cx"]
    cy = box["cy"]
    r = max(8.0, min(box["w"], box["h"]) / 2.5)
    return cx, cy, r


# HoughCircles 的工作边长上限。超过则先把子图降采样、出结果再放大回去。
# 必要性（实测）: HoughCircles 代价 ∝ 子图面积 × 半径档数, 当 YOLO 给出近全画幅的
# 大框时, 半径档数随之膨胀到几百档, 耗时爆炸:
#   24.jpg  box 1440x1503  -> 81.0 s
#   超压1   box 2042x1862  -> 13.0 s
# 而正常尺寸的框只要零点几秒（21.jpeg box 457x446 -> 0.20 s）。
# 网页端因此会在"检测框偏大"时每张图卡近 100 秒。降采样上限取的 800 是实测
# 折中: 超压1 在 cap=800 下得到 r=862, 与原始尺度的 r=861 完全一致;
# 而 cap=640 会给出 r=766（偏小 11%）—— 半径偏小会让 gauge_3stage 的取样环带
# 内缩、丢掉外圈色带, 所以不能再往下压。
HOUGH_MAX_SIDE = 800


def hough_circle(img_bgr, box: dict, scale=1.25, max_side=HOUGH_MAX_SIDE):
    """② HoughCircles: 限圆心邻域与半径范围, 取与先验最接近者

    max_side: 子图长边降采样上限（0/None = 不降采样, 原始尺度, 很慢）。
    """
    w0, h0 = box["w"], box["h"]
    cx0, cy0, r0 = circle_prior(box)
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(img_bgr.shape[1], int(box["x2"]))
    y2 = min(img_bgr.shape[0], int(box["y2"]))
    sub = img_bgr[y1:y2, x1:x2]
    if sub.size == 0 or min(sub.shape[:2]) < 24:
        return None
    # 大框先降采样, 把 Hough 的工作量按面积平方级压下来
    sh, sw = sub.shape[:2]
    k = 1.0                                   # 子图 -> 工作图 的缩放因子
    if max_side and max(sh, sw) > max_side:
        k = max_side / float(max(sh, sw))
        sub = cv2.resize(sub, (max(24, int(sw * k)), max(24, int(sh * k))),
                         interpolation=cv2.INTER_AREA)
    # 半径实际范围（换算到工作尺度）
    maxr = r0 * scale
    minr = r0 * 0.72
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,
                               dp=1.2, minDist=maxr * 2 * k,
                               param1=80, param2=32,
                               minRadius=max(1, int(minr * k)),
                               maxRadius=max(2, int(maxr * k)))
    if circles is None:
        return None
    best, best_score = None, -1.0
    for c in circles[0]:
        ccx = c[0] / k + x1                   # 结果换算回原图尺度
        ccy = c[1] / k + y1
        rr = c[2] / k
        d_center = np.hypot(ccx - cx0, ccy - cy0) / max(r0, 1e-6)
        d_radius = abs(rr - r0) / r0
        # 圆心越近越好(权重大), 半径越接近越好
        score = -(2.0 * d_center + 0.6 * d_radius)
        if d_center < 0.25 and d_radius < 0.35 and score > best_score:
            best = (ccx, ccy, rr)
            best_score = score
    return best


def white_dial_circle(img_bgr, box: dict, scale=1.25):
    """③ 白盘面连通域 minEnclosingCircle"""
    w0, h0 = box["w"], box["h"]
    cx0, cy0, r0 = circle_prior(box)
    x1 = max(0, int(box["x1"])); y1 = max(0, int(box["y1"]))
    x2 = min(img_bgr.shape[1], int(box["x2"]))
    y2 = min(img_bgr.shape[0], int(box["y2"]))
    sub = img_bgr[y1:y2, x1:x2]
    sh, sw = sub.shape[:2]
    if sh < 24 or sw < 24:
        return None
    hsv = cv2.cvtColor(sub, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)
    # 白盘面: 高亮 + 低饱和 (比之前 fit_dial_ellipse 放宽)
    white = ((V > 120) & (S < 90)).astype(np.uint8) * 255
    # 只保留中央 60%: 外壳/背景大多在边缘
    yy, xx = np.indices((sh, sw))
    central = (np.abs(xx - sw / 2) < sw * 0.30) & \
              (np.abs(yy - sh / 2) < sh * 0.30)
    white = (white > 0) & central
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    white = cv2.morphologyEx(white.astype(np.uint8) * 255,
                             cv2.MORPH_CLOSE, k, iterations=2)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    if n < 2:
        return None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    if stats[idx, cv2.CC_STAT_AREA] < (sw * sh * 0.03):
        return None
    mask = (labels == idx).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    big = max(cnts, key=cv2.contourArea)
    if len(big) < 20:
        return None
    (ccx, ccy), rr = cv2.minEnclosingCircle(big)
    ccx += x1
    ccy += y1
    # 合理性: 中心偏离先验中心过远或半径异常则弃用
    if np.hypot(ccx - cx0, ccy - cy0) > 0.30 * r0:
        return None
    if not (r0 * 0.65 <= rr <= r0 * scale):
        return None
    return ccx, ccy, rr


# ---------------- ④ 射线法 (唯一不受检测框尺寸约束的半径估计) ----------------

RAY_ANGLES = 180       # 射线条数
RAY_RUN = 4            # 判定"盘面"所需的连续亮像素数(px, 源图尺度)
RAY_V_FACE = 85        # 盘面像素亮度下限
RAY_V_DARK = 70        # 盘外像素亮度上限
RAY_LO = 0.35          # 径向搜索范围(×r0), 下界要够小以容住偏移的盘心
RAY_HI = 1.70          # 上界要明显超出框短边的一半, 否则又变成单边截断
RAY_MIN_HITS = 30      # 命中射线数下限, 低于此判为不可用(退回旧方法)
RAY_QUANTILE = 50      # 命中半径取中位数: 指针/水印/反光只会污染个别射线
RAY_MAX_OVER = 2.2     # 半径超过 2.2*r0 判为被外部亮区(白墙/标签)骗到, 弃用


def ray_circle(img_bgr, box: dict, lo=RAY_LO, hi=RAY_HI, quantile=RAY_QUANTILE):
    """④ 射线法: 沿 180 条射线从中心往外找第一个"盘面(亮)->盘外(暗)"边界。

    思路: 盘面不论白底还是彩色底, 亮度都显著高于其外侧的暗环/背景。于是从中心
    沿各方向扫描, 取"从外往内第一段长度 >= RAY_RUN 的亮区"的起点(即亮段与外侧
    暗区的交界)作为该方向的盘面外沿, 最后取所有命中方向的中位数。中位数而不是
    最大值, 是为了让指针、水印、盘面反光这些只污染个别射线的因素失效。

    为什么必须补这一条(而不是继续调 Hough):
      · hough_circle 在 bbox 子图上跑, 半径窗被结构性钉在 [0.72, 1.25]*r0,
        上限恰等于框短边的一半。检测框只要贴紧表盘, 真实半径就落在搜索窗之外,
        再怎么调 param2 / dp 都够不到。
      · white_dial_circle 的掩膜是"高亮 + 低饱和"(V>120 & S<90), 彩色色带(红/黄)
        被直接排除在掩膜之外, minEnclosingCircle 只能圈住白色盘心。
      两条路都是**单边偏小**, 而半径偏小会让 crop_scale = 476/r 偏大, 盘面外圈
      的色带被推出画幅切掉 —— 而色带正是第二步分类器判"正常/欠压/超压"的依据。

    射线法在整幅原图上工作, 不受框尺寸约束, 因此不受上述限制。圆心沿用检测框
    中心(YOLO 松框标定出的中心是可信的), 本函数只解"半径够不到"这一个失效。

    返回 (cx, cy, r) 或 None(命中不足 / 半径离谱)。
    """
    cx, cy, r0 = circle_prior(box)
    gray = cv2.GaussianBlur(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    H, W = gray.shape
    rs = np.arange(lo * r0, hi * r0, 1.0)
    if rs.size < 8:
        return None
    # 一次性算出所有射线上的采样点(180×~千点), 避免 Python 双层循环
    ang = np.linspace(0.0, 2.0 * np.pi, RAY_ANGLES, endpoint=False)
    xs = np.rint(cx + np.cos(ang)[:, None] * rs[None, :]).astype(np.int32)
    ys = np.rint(cy + np.sin(ang)[:, None] * rs[None, :]).astype(np.int32)
    inside = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
    prof = np.full(xs.shape, -1.0, np.float32)   # 出界记 -1: 必然判为"暗"
    prof[inside] = gray[ys[inside], xs[inside]]
    bright = prof > RAY_V_FACE

    n = rs.size
    hits = []
    for k in range(RAY_ANGLES):
        idx = np.flatnonzero(bright[k])
        if idx.size == 0:
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)          # 亮段的断开处
        seg_s = np.r_[idx[0], idx[brk + 1]]             # 各亮段起点
        seg_e = np.r_[idx[brk], idx[-1]]                # 各亮段终点
        # 从最外侧的亮段开始判定: 长度够 + 该段外侧确实变暗 => 这就是盘面外沿
        for s, e in zip(seg_s[::-1], seg_e[::-1]):
            if (e - s + 1) >= RAY_RUN and \
                    (e + 1 >= n or prof[k, e + 1] <= RAY_V_DARK):
                hits.append(rs[s])
                break
    if len(hits) < RAY_MIN_HITS:
        return None
    r = float(np.percentile(np.asarray(hits, dtype=np.float64), quantile))
    # 上界防呆: 盘面外侧若是一大片亮墙/说明标签, 会被误判成盘面而给出离谱大圆
    if not (r > 0) or r > r0 * RAY_MAX_OVER:
        return None
    return cx, cy, r


def refine_circle(img_bgr, box: dict, scale=1.25, max_side=HOUGH_MAX_SIDE):
    """融合四种方法, 返回 (cx, cy, r, method)。绝对不失败: 至少返回几何先验

    分工(2026-09-15 重写):
      圆心 —— 取 hough / white_dial 里与先验中心最贴合且过门限者, 只有这两个
              方法有真实的圆心估计; 都不可用时退回先验中心。射线法不参与圆心,
              避免在修半径的同时引入新的不确定性。
      半径 —— 取射线法, 并与先验 r0 取上包络。

    为什么半径取上包络: 半径偏小会让 crop_scale = 476/r 偏大, 盘面外圈色带被
    推出画幅切掉, 信息不可逆丢失; 半径偏大只是多带一圈背景。代价不对称, 所以
    旧版 "r > r0 就额外扣 0.3 分" 的**宁小勿大偏置已被删除**。
    射线法不可用时退化为 max(r0, 旧方法半径) —— 旧方法单边偏小, 先验做下限
    至少能消掉"比先验还小"的那部分误差。

    max_side: 透传给 hough_circle 的降采样上限（0 = 原始尺度, 用于做对照）。
    """
    cx0, cy0, r0 = circle_prior(box)
    c_h = hough_circle(img_bgr, box, scale, max_side=max_side)
    c_w = white_dial_circle(img_bgr, box, scale)

    # ---- ① 圆心 ----
    cands = [(c, m) for c, m in ((c_h, "hough"), (c_w, "white_dial"))
             if c is not None and
             np.hypot(c[0] - cx0, c[1] - cy0) <= 0.30 * r0]
    if cands:
        (ccx, ccy, cr), cm = min(
            cands, key=lambda t: np.hypot(t[0][0] - cx0, t[0][1] - cy0))
    else:
        ccx, ccy, cr, cm = cx0, cy0, r0, "prior"

    # ---- ② 半径 ----
    c_r = ray_circle(img_bgr, box)
    if c_r is not None:
        r = max(r0, c_r[2])
        rm = "ray"
    else:
        r = max(r0, float(cr))
        rm = cm
    # method 形如 "ray+hough" = 半径取自射线法、圆心取自 hough;
    # 不以 "ray" 开头则说明射线法没命中(或半径离谱被否), 半径退回旧方法并用
    # 先验 r0 兜底 —— 配合 circle_info.json 里的 r_prior 可反推兜底幅度。
    method = rm if rm == cm else f"{rm}+{cm}"
    return float(ccx), float(ccy), float(r), method


# ---------------- 裁切与可视化 ----------------

# 上采样插值: LANCZOS4 的 4 瓣 sinc 核比 CUBIC 的边缘保持更好, 对指针细线、
# 刻度文字这类高频结构明显更利落(代价是极强边缘会有轻微振铃)。
# 降采样保持 INTER_AREA 做面积平均抗锯齿。
INTERP_UP = cv2.INTER_LANCZOS4
INTERP_DOWN = cv2.INTER_AREA

# --interp 的可选值
INTERP_CHOICES = {"lanczos4": cv2.INTER_LANCZOS4, "cubic": cv2.INTER_CUBIC,
                  "linear": cv2.INTER_LINEAR, "area": cv2.INTER_AREA,
                  "nearest": cv2.INTER_NEAREST}

# 放大倍率低于此值时不锐化 —— 此时插值本身基本不引入可见模糊, 锐化只会有害。
SHARPEN_MIN_SCALE = 1.5


def unsharp(img, amount: float, sigma: float):
    """Unsharp Mask: I + amount * (I - Gauss(I, sigma))。

    刻意不用 cv2.addWeighted 的 beta 参数直接给负权重组合, 而是先算出
    模糊层再叠加, 这样 amount=0 时能严格等于原图(不做无谓的浮点往返)。
    """
    if amount <= 0 or sigma <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def auto_sharpen(scale: float):
    """按放大倍率给出 (amount, sigma); 不需要锐化时返回 (0.0, 0.0)。

    参数标定来自 tools/compare_upscale.py 的实测(21/28/24 三张真实留出图,
    4.45x~8.35x 上采样, 逐档扫 amount∈{0.4,0.7,1.0} × sigma∈{1.0,1.5,2.0},
    指标 = 圆内 Tenengrad 锐度 / 越界能量% / 相对基线偏移 p99.9 / 圆内过曝%):
        sigma 2.0 起出现肉眼可见的边缘光晕, amount>0.7 时盘面环纹出现假亮带;
        amount 0.7 / sigma 1.5 在"锐度 +10%"与"过曝 +41%"之间取得平衡,
        是所有测试图上唯一既明显变锐又不引入可见伪影的档位。

    sigma 为什么基本恒定、只做温和增长:
        cubic / lanczos4 这类插值核的支撑宽度是**输出像素固定**的(cubic 4px、
        lanczos4 8px), 所以插值本身引入的软化宽度与放大倍率无关 —— 这是主项,
        对应常数 sigma。次要项是源图自带的镜头/运动模糊, 它会被放大 s 倍,
        对应线性增长。两者合起来就是下面这条温和的线性式(实测区间内 ≈1.4~1.6)。
        (早期版本写成 sigma = 0.35*scale 是错的, 在 8 倍下给到 2.9, 实测光晕明显。)
    """
    if scale < SHARPEN_MIN_SCALE:
        return 0.0, 0.0
    amount = min(0.70, 0.30 * math.log2(scale))
    sigma = min(1.8, 1.2 + 0.05 * scale)
    return amount, sigma


def parse_sharpen(s):
    """CLI 参数 -> crop_circle 的 sharpen 实参。

    "auto" -> None(按倍率自动) | "off" -> False(关闭)
    "0.7"  -> 只给 amount, sigma 由倍率推 | "0.7,1.5" -> 完全手控
    """
    t = str(s).strip().lower()
    if t in ("auto", ""):
        return None
    if t in ("off", "none", "no"):
        return False
    vals = [float(p) for p in t.replace("/", ",").split(",") if p.strip()]
    if not vals:
        return None
    return vals[0] if len(vals) == 1 else (vals[0], vals[1])


def crop_circle(img_bgr, cx, cy, r, pad=4, fill=0, out_size=None,
                interp=None, sharpen=None):
    """圆形裁切: 圆外填 fill(0黑/255白)

    out_size=None  原生尺寸输出。边长 = 2r+2pad, 圆心在图像正中, 圆半径 r,
                   四周留 pad 像素填充带。(与旧行为逐像素一致)
    out_size=S     输出恒为 S×S, 且**保持 pad 的绝对像素宽度**:
                   目标半径 = S/2 - pad。这一点很关键 ——
                   tools/gauge_3stage.py:292 里写死了 `R = w/2 - 4`
                   (注释"crop_circle 的 pad=4"), 只要目标半径满足该式,
                   任意 S 下三段式的取样环带(0.55R~0.985R)都仍然精确对齐,
                   下游无需任何改动。
                   实现上直接对源图做一次仿射重采样(而非"先裁后缩放"),
                   少一次插值, 保住原生像素细节。

    interp   None  自动: 放大用 INTER_LANCZOS4, 缩小用 INTER_AREA
             也可传 cv2.INTER_* 常量强制指定(如 INTER_CUBIC 复现旧行为)
    sharpen  None  自动: 按放大倍率决定(见 auto_sharpen), 小倍率不锐化
             False 关闭锐化
             float 只给 amount, sigma 由倍率推出
             (amount, sigma) 完全手控

    为什么需要锐化: 表盘在原图里往往只有 r=57~130px, 放大到目标半径 476 是
    4~8 倍上采样。放大不产生新像素, 插值只会把低频软化铺开, 表现为指针细线
    和刻度发虚。USM 不是"恢复"细节, 而是把被插值抹平的边缘斜率重新拉陡,
    让下游(分类网络 / 三段式取样环带)拿到更接近原始反差的输入。
    锐化放在圆外填充之前 —— 此时方块里还是完整的重采样图像内容, 不会像
    "先填黑再锐化"那样在圆周内侧拉出一圈假的高亮环。
    """
    if out_size:
        S = int(out_size)
        target_r = S / 2.0 - float(pad)
        if target_r <= 0:
            raise ValueError(f"out_size={S} 与 pad={pad} 不兼容(目标半径<=0)")
        scale = target_r / float(r)          # 源半径 r -> 目标半径 target_r
        inv = 1.0 / scale
        c = (S - 1) / 2.0                    # 与 gauge_3stage.analyze_crop 同口径
        M = np.array([[inv, 0.0, cx - c * inv],
                      [0.0, inv, cy - c * inv]], dtype=np.float64)
        if scale < 1.0:
            # 降采样恒用 INTER_AREA(面积平均): 目标半径比源半径小时, 面积平均是
            # 无条件正确的抗锯齿做法, 所以 --interp 只用来选放大方向的核,
            # 不允许用户把它改成 nearest 之类而丢掉抗锯齿。
            interp = INTERP_DOWN
        elif interp is None:
            interp = INTERP_UP
        # 注意: warpAffine 默认把 M 当"源->目标"正向变换并内部求逆,
        # 而这里 M 是"目标->源"的采样矩阵, 必须加 WARP_INVERSE_MAP 才不会反向。
        out = cv2.warpAffine(img_bgr, M, (S, S),
                             flags=interp | cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_CONSTANT,
                             borderValue=(fill, fill, fill))

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

        xx, yy = np.meshgrid(np.arange(S), np.arange(S))
        outside = (xx - c) ** 2 + (yy - c) ** 2 > (target_r - 1) ** 2
        out[outside] = fill
        return out

    side = int(2 * r + 2 * pad)
    out = np.full((side, side, 3), fill, np.uint8)
    xx, yy = np.meshgrid(np.arange(side), np.arange(side))
    d2 = (xx - (r + pad)) ** 2 + (yy - (r + pad)) ** 2
    in_c = d2 <= (r - 1) ** 2
    # 映射回原图采样
    xs = (xx[in_c] + cx - (r + pad)).astype(np.int32)
    ys = (yy[in_c] + cy - (r + pad)).astype(np.int32)
    ok = (xs >= 0) & (xs < img_bgr.shape[1]) & \
         (ys >= 0) & (ys < img_bgr.shape[0])
    out[yy[in_c][ok], xx[in_c][ok]] = \
        img_bgr[ys[ok], xs[ok]]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--outdir", required=True)
    ap.add_argument("--yolo", default=DEFAULT_YOLO)
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--scale", type=float, default=1.25,
                    help="相对先验半径的搜索上限")
    ap.add_argument("--size", type=int, default=960,
                    help="圆形裁切输出边长, 默认 960(目标半径=size/2-pad, "
                         "保证 gauge_3stage 的 R=w/2-4 成立); 0=原生尺寸")
    ap.add_argument("--imgsz", type=int, default=0,
                    help="YOLO 推理分辨率; 0=沿用 checkpoint 训练分辨率(默认, 推荐)")
    ap.add_argument("--fill", choices=["black", "white"], default="black")
    ap.add_argument("--interp", default="auto",
                    choices=["auto"] + sorted(INTERP_CHOICES),
                    help="上采样插值。auto(默认)=放大用 lanczos4 / 缩小用 area; "
                         "cubic 可复现 2026-09-14 之前的旧观感")
    ap.add_argument("--sharpen", default="auto",
                    help="圆内 USM 锐化: auto(默认, 按放大倍率取 amount/sigma) | "
                         "off | 单个 amount(如 0.7) | amount,sigma(如 0.7,1.5)")
    ap.add_argument("--overlay", default=None,
                    help="叠加可视化目录(画矩形/圆对比)")
    args = ap.parse_args()

    if not os.path.exists(args.yolo):
        sys.exit(f"YOLO 权重不存在: {args.yolo}")
    from ultralytics import YOLO
    yolo = YOLO(args.yolo)

    if os.path.isdir(args.input):
        paths = sorted(
            p for p in glob.glob(os.path.join(args.input, "*"))
            if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")))
    else:
        paths = [args.input]

    os.makedirs(args.outdir, exist_ok=True)
    if args.overlay:
        os.makedirs(args.overlay, exist_ok=True)
    fill = 0 if args.fill == "black" else 255
    interp = None if args.interp == "auto" else INTERP_CHOICES[args.interp]
    sharpen = parse_sharpen(args.sharpen)
    rows = []
    print(f"{'image':>22s}  {'method':<14s}  "
          f"{'circle(cx,cy,r)':>22s}   conf")
    for p in paths:
        img = imread_any(p)
        if img is None:
            continue
        kw = {"imgsz": args.imgsz} if args.imgsz else {}
        res = yolo.predict(img, conf=args.conf, verbose=False, **kw)
        name = os.path.splitext(os.path.basename(p))[0]
        if not res or res[0].boxes is None or len(res[0].boxes) == 0:
            print(f"{name:>22s}  未检出")
            continue
        b = res[0].boxes[0]
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        box = {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
               "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2,
               "w": x2 - x1, "h": y2 - y1, "conf": float(b.conf[0])}
        cx, cy, r, method = refine_circle(img, box, scale=args.scale)
        crop = crop_circle(img, cx, cy, r, fill=fill,
                           out_size=(args.size or None),
                           interp=interp, sharpen=sharpen)
        out_p = os.path.join(args.outdir, name + ".jpg")
        imwrite_any(out_p, crop)
        rows.append({"file": name,
                     "box": {k: float(v) for k, v in box.items()},
                     "cx": float(cx), "cy": float(cy),
                     "r": float(r), "method": method,
                     "conf": float(box["conf"]),
                     "imgsz": args.imgsz,
                     # 几何先验半径 = min(w,h)/2.5。射线法没命中时 r 会被它托底,
                     # 记下来才能事后分辨"r 是射线法测得"还是"被先验下限截断"。
                     "r_prior": round(float(circle_prior(box)[2]), 2),
                     "crop_size": [int(crop.shape[1]), int(crop.shape[0])],
                     "crop_pad": 4,
                     # r 在 hough/white_dial 分支是 numpy.float32, round() 会原样
                     # 返回 np.float32, 导致最后的 json.dump 抛
                     # "Object of type float32 is not JSON serializable"。
                     # 必须先转成 Python float 再 round。
                     "crop_scale": round(float((args.size / 2.0 - 4) / r), 3)
                     if args.size else 1.0,
                     "interp": args.interp,
                     "sharpen": args.sharpen})
        print(f"{name:>22s}  {method:<14s}  "
              f"({cx:5.1f},{cy:5.1f},{r:4.1f})  {box['conf']:.2f}")

        if args.overlay:
            vis = img.copy()
            # YOLO 矩形
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                          (255, 255, 0), 2)
            # 精化圆
            cv2.circle(vis, (int(cx), int(cy)), int(r), (0, 200, 255), 2)
            cv2.circle(vis, (int(cx), int(cy)), 3, (0, 0, 255), -1)
            cv2.putText(vis, f"rect vs circle({method})", (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            imwrite_any(os.path.join(args.overlay, name + ".jpg"), vis)
    import json
    json.dump(rows, open(os.path.join(args.outdir, "circle_info.json"),
                         "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"\n完成: {len(rows)} 张圆形裁切 -> {args.outdir}")


if __name__ == "__main__":
    main()
