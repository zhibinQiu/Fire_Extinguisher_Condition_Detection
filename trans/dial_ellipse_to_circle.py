# -*- coding: utf-8 -*-
"""
椭圆形表盘检测 + 校正成正圆

流程：
  S1 预处理：等比缩小（高度->600，保持长宽比） + 3x3 高斯平滑
  S2 表盘定位：
     阶段1 Canny -> 轮廓 -> fitEllipse 候选（边缘覆盖率>=0.5 排除弧线假椭圆）
           -> 协方差白化 Σ^{-1/2} 仿射校正为圆 -> 红黄绿三色 HSI 圆周验证
     阶段2 兜底：Hough 圆变换（正圆=椭圆特例），同样做三色验证

用法：
    python dial_ellipse_to_circle.py            # 处理下方 IMG_PATH 指定的图
    python dial_ellipse_to_circle.py xxx.jpg    # 处理命令行指定的图

输出（与输入图同目录，与原图同分辨率）：
    *_ellipse_result.jpg  原图 + 拟合椭圆/圆标注
    *_circle.jpg          校正后的正圆表盘（白化校正作用于原图，圆心固定在图像正中心，
                          半径=(ax1+ax2)/2；圆外区域用常量 (30,30,30) 填充）
检测在 S1 预处理后的 600 高工作图上进行（专利参数的前提），
检出参数按缩放比映射回原图坐标系后再做校正与标注。

注意（与 gauge_recognizer.py 相同的实战经验）：
    warpAffine 默认对 M 求逆后按 dst->src 采样，M 应传 src->dst 前向矩阵，
    【不要】加 WARP_INVERSE_MAP（加 flag 反而会得到未校正的椭圆）。
"""
import math
import cv2
import numpy as np
import sys
import os
import time

# ============ 配置：换图只改这一行（或命令行传参） ============
# 本精简包不内置示例图，默认留空；CLI 必须显式给路径。
# （在线流水线不走这里：server.py 直接调 correct_to_circle()。
#   主工程里这一行原本硬编码指向 device_ocr/labeling/... 的一张示例图，
#   提炼时去掉了这层对包外绝对路径的依赖。）
IMG_PATH = ""


# ----------------------------- 参数（专利给定值 + 真实照片实测微调，同 gauge_recognizer） -----------------------------
class Params:
    # S1
    resize_height = 600          # 尺寸变换后的图像高度（专利示例值）
    gaussian_ksize = 3           # 高斯平滑模板 3x3

    # S2 - Hough 圆兜底
    hough_min_radius = 100       # 表盘区间 rmin（专利优选值，像素）
    hough_max_radius = 200       # 表盘区间 rmax

    # S2 - 三色验证（HSI 颜色判据，H: 0-360 度, S: 0-1）
    scan_radius_ratios = (0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9)
    scan_step_deg = 0.5          # 扫描角度间隔 0.5 度 -> 720 点
    red_h_min = 330.0            # 红: H>330, S>0.12
    red_s_min = 0.12
    green_h_range = (100.0, 200.0)  # 绿: 100<H<200, S>0.08
    green_s_min = 0.08
    yellow_h_range = (38.0, 100.0)  # 黄: 38<H<100, S>0.12
    yellow_s_min = 0.12
    dial_min_count_each = 8      # 红/绿/黄扫描点数量各自需超过的数量

    # S2 - 圆心可信度（在线流水线用）
    # 送进来的图是「第一步检测框裁切 + resize」来的，框已经把表盘框住了，
    # 所以**表盘圆心必然落在图像中心附近**。检测到的圆心偏出去太多 = 算法
    # 找错了东西（实测 10__原图 的 Hough 兜底把圆心定到 0.23·短边之外，
    # 校正图里表盘被推出画面，第二步针尖误差从 2px 飙到 215px）。
    # 超过这个比例就判定「校正不可信」，退回未校正的直裁图 ——
    # 宁可少校正几张，也不要交一张错的校正图。
    max_offcenter_ratio = 0.15

    # S2 - 椭圆检测与校正
    ellipse_min_ratio = 0.35     # 短轴/长轴 下限（过扁视为非表盘）
    ellipse_max_candidates = 10  # 最多尝试的椭圆候选数
    ellipse_min_support = 50     # 拟合椭圆所需的最少轮廓点数
    ellipse_edge_support = 0.5   # 椭圆采样点落在边缘上的比例下限


COLOR_RED = "R"
COLOR_GREEN = "G"
COLOR_YELLOW = "Y"
COLOR_OTHER = "E"


# ----------------------------- 颜色工具（同 gauge_recognizer） -----------------------------
def rgb_to_hsi(bgr):
    """按专利公式将单个 BGR 像素转换为 HSI。返回 (H: 0-360 度, S: 0-1, I: 0-1)。"""
    b, g, r = [float(v) for v in bgr]
    i = (r + g + b) / 3.0
    total = r + g + b
    if total <= 0:
        return 0.0, 0.0, 0.0
    s = 1.0 - 3.0 * min(r, g, b) / total
    num = 0.5 * ((r - g) + (r - b))
    den = math.sqrt((r - g) ** 2 + (r - b) * (g - b))
    if den <= 1e-9:
        h = 0.0
    else:
        cos_v = max(-1.0, min(1.0, num / den))
        h = math.degrees(math.acos(cos_v))
    if b > g:
        h = 360.0 - h
    # 专利：红色 H 分布在 0-20 与 330-360，为保证连续，H<20 时转为 360-H
    if h < 20.0:
        h = 360.0 - h
    return h, s, i / 255.0


def classify_zone_color(h, s):
    """按 HSI 判据分类表盘颜色：返回 R/G/Y/E。"""
    if h > Params.red_h_min and s > Params.red_s_min:
        return COLOR_RED
    if Params.green_h_range[0] < h < Params.green_h_range[1] and s > Params.green_s_min:
        return COLOR_GREEN
    if Params.yellow_h_range[0] < h < Params.yellow_h_range[1] and s > Params.yellow_s_min:
        return COLOR_YELLOW
    return COLOR_OTHER


def sample_hsi(img, cx, cy, r, theta_deg):
    """在圆周上取一点（3x3 中值滤波抗噪），返回 (h, s, i, x, y)。"""
    rad = math.radians(theta_deg)
    x = int(round(cx + r * math.cos(rad)))
    y = int(round(cy + r * math.sin(rad)))
    h_img, w_img = img.shape[:2]
    x = max(1, min(w_img - 2, x))
    y = max(1, min(h_img - 2, y))
    patch = img[y - 1:y + 2, x - 1:x + 2].reshape(-1, 3)
    bgr = np.median(patch, axis=0)
    h, s, i = rgb_to_hsi(bgr)
    return h, s, i, x, y


# ----------------------------- S1 预处理（同 gauge_recognizer） -----------------------------
def preprocess(img):
    """尺寸变换（等比缩小到高 600，保持长宽比） + 3x3 高斯平滑。"""
    h_src, w_src = img.shape[:2]
    scale = Params.resize_height / float(h_src)
    w_dst = int(round(w_src * scale))
    if scale < 1.0:
        resized = cv2.resize(img, (w_dst, Params.resize_height), interpolation=cv2.INTER_AREA)
    else:
        resized = img.copy()
    blurred = cv2.GaussianBlur(resized, (Params.gaussian_ksize, Params.gaussian_ksize), 0)
    return blurred, scale


# ----------------------------- S2 表盘定位（同 gauge_recognizer） -----------------------------
def _verify_dial(img, cx, cy, rc):
    """多半径圆周扫描（每圈 720 点），每种颜色取各半径中的最大计数。
    返回 (是否表盘, 三色计数, 最佳扫描半径)。"""
    n = int(round(360.0 / Params.scan_step_deg))
    best = {COLOR_RED: 0, COLOR_GREEN: 0, COLOR_YELLOW: 0}
    best_r, best_total = Params.scan_radius_ratios[0] * rc, -1
    for ratio in Params.scan_radius_ratios:
        r_scan = ratio * rc
        counts = {COLOR_RED: 0, COLOR_GREEN: 0, COLOR_YELLOW: 0}
        for k in range(n):
            theta = k * Params.scan_step_deg
            h, s, _, _, _ = sample_hsi(img, cx, cy, r_scan, theta)
            c = classify_zone_color(h, s)
            if c in counts:
                counts[c] += 1
        for c in best:
            best[c] = max(best[c], counts[c])
        total = sum(counts.values())
        if total > best_total:
            best_total, best_r = total, r_scan
    ok = all(v >= Params.dial_min_count_each for v in best.values())
    return ok, best, best_r


def _ellipse_candidates(img):
    """Canny 边缘 -> 轮廓 -> fitEllipse，筛选可能的表盘椭圆候选。
    返回按支撑点数降序的候选列表（dict: cx, cy, ax1, ax2, ang, r_out, ratio, support）。
    ax1/ax2 为 fitEllipse 原始半轴长（与 ang 保持同一约定），r_out 为校正后的圆半径。"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    h_img, w_img = img.shape[:2]
    cands = []
    for cnt in contours:
        if len(cnt) < Params.ellipse_min_support:
            continue
        try:
            (ex, ey), (d1, d2), ang = cv2.fitEllipse(cnt)
        except cv2.error:
            continue
        ax1, ax2 = d1 / 2.0, d2 / 2.0
        a, b = max(ax1, ax2), min(ax1, ax2)
        if a < 30 or a > 0.9 * max(h_img, w_img):
            continue
        ratio = b / a
        if ratio < Params.ellipse_min_ratio:
            continue
        if not (0 <= ex <= w_img and 0 <= ey <= h_img):
            continue
        cands.append({"cx": float(ex), "cy": float(ey), "ax1": float(ax1), "ax2": float(ax2),
                      "ang": float(ang), "r_out": (ax1 + ax2) / 2.0,
                      "ratio": float(ratio), "support": len(cnt)})
    # 按支撑点降序 + 去重（圆心/长轴接近的视为同一椭圆）
    cands.sort(key=lambda e: -e["support"])
    kept = []
    for e in cands:
        dup = any(abs(e["cx"] - k["cx"]) < 15 and abs(e["cy"] - k["cy"]) < 15
                  and abs(e["r_out"] - k["r_out"]) < 15 for k in kept)
        if not dup:
            kept.append(e)
    # 几何过滤：椭圆整周采样点的边缘覆盖率（排除弧线拟合的假椭圆）
    dil = cv2.dilate(edges, np.ones((3, 3), np.uint8))
    passed = []
    for e in kept:
        poly = cv2.ellipse2Poly((int(round(e["cx"])), int(round(e["cy"]))),
                                (max(1, int(round(e["ax1"]))), max(1, int(round(e["ax2"])))),
                                int(round(e["ang"])), 0, 360, 5)
        xs = np.clip(poly[:, 0], 0, w_img - 1)
        ys = np.clip(poly[:, 1], 0, h_img - 1)
        cover = float((dil[ys, xs] > 0).mean())
        e["edge_cover"] = cover
        if cover >= Params.ellipse_edge_support:
            passed.append(e)
        if len(passed) >= Params.ellipse_max_candidates:
            break
    return passed


def _ellipse_to_circle_matrix(e):
    """构造把椭圆映射为圆的仿射矩阵（src -> dst）。
    用 cv2.ellipse2Poly 在与 fitEllipse 相同的参数约定下采样椭圆轮廓点，
    对采样点求协方差 Σ，白化变换 W=Σ^{-1/2} 把椭圆映为圆（白化半径 sqrt(2)），
    再缩放到目标半径 r_out。全程数值计算，规避 fitEllipse 角度约定的坑。"""
    cx, cy = e["cx"], e["cy"]
    poly = cv2.ellipse2Poly((int(round(cx)), int(round(cy))),
                            (max(1, int(round(e["ax1"]))), max(1, int(round(e["ax2"])))),
                            int(round(e["ang"])), 0, 360, 5)
    pts = poly.astype(np.float64)
    cov = np.cov((pts - pts.mean(axis=0)).T)
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-9)
    W = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T      # Σ^{-1/2}
    s = e["r_out"] / math.sqrt(2.0)                   # 白化半径 sqrt(2) -> r_out
    c = np.array([cx, cy])
    M = np.hstack([s * W, (c - s * W @ c).reshape(2, 1)])
    return M.astype(np.float64)


def find_dial(img):
    """表盘定位：先识别椭圆型表盘 -> 仿射变换为圆形表盘 -> 三色验证；
    Hough 圆路径作为兜底（正圆是椭圆的特例）。
    返回 dict(img=校正后工作图, cx, cy, r, counts, best_r, ellipse, M, corrected)；未找到返回 None。"""
    # 阶段 1：椭圆检测 -> 校正为圆 -> 三色验证
    for e in _ellipse_candidates(img):
        M = _ellipse_to_circle_matrix(e)
        # M 为 src->dst 映射；warpAffine 默认对 M 求逆后按 dst->src 采样，
        # 因此这里【不要】加 WARP_INVERSE_MAP（实测加 flag 反而会得到未校正的椭圆）
        warped = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REPLICATE)
        ok, counts, best_r = _verify_dial(warped, e["cx"], e["cy"], e["r_out"])
        if ok:
            return {"img": warped, "cx": e["cx"], "cy": e["cy"], "r": e["r_out"],
                    "counts": counts, "best_r": best_r, "ellipse": e, "M": M,
                    "corrected": True}

    # 阶段 2（兜底）：Hough 圆变换（正圆=椭圆特例，正拍时更快命中）
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
        param1=100, param2=60,
        minRadius=Params.hough_min_radius, maxRadius=Params.hough_max_radius)
    if circles is not None:
        for c in circles[0]:
            cx, cy, rc = float(c[0]), float(c[1]), float(c[2])
            ok, counts, best_r = _verify_dial(img, cx, cy, rc)
            if ok:
                return {"img": img, "cx": cx, "cy": cy, "r": rc,
                        "counts": counts, "best_r": best_r, "ellipse": None, "M": None,
                        "corrected": False}

    circles2 = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=80,
        param1=100, param2=50,
        minRadius=max(30, Params.hough_min_radius // 3),
        maxRadius=Params.hough_max_radius * 2)
    if circles2 is not None:
        for c in circles2[0]:
            cx, cy, rc = float(c[0]), float(c[1]), float(c[2])
            ok, counts, best_r = _verify_dial(img, cx, cy, rc)
            if ok:
                return {"img": img, "cx": cx, "cy": cy, "r": rc,
                        "counts": counts, "best_r": best_r, "ellipse": None, "M": None,
                        "corrected": False}
    return None


# ----------------------------- 可复用 API（给在线流水线调用） -----------------------------
def correct_to_circle(img_src):
    """把一张（已经裁好的）表盘图校正成正圆。**不落盘、不打印**，纯函数。

    给 webapp_v2 的流水线用：第一步 YOLO 检出表盘并裁出 patch 后，先过这里
    变成正圆，再送第二步 pose 模型。

    为什么值得加这一步
    ------------------
    侧视/斜拍时表盘在图像上是**椭圆**。圆心 -> 针尖的连线在这张图上与各条
    边界连线的夹角，都被透视**非等比压缩过** —— 同一个真实角度，在椭圆的长轴
    方向与短轴方向量出来不一样大。三态判据本质是比角度，所以不校正就会偏。
    白化校正把椭圆映成正圆后，角度才回到真实比例。

    Parameters
    ----------
    img_src : ndarray  BGR，任意分辨率（通常是 960x960 的检测框直裁图）

    Returns
    -------
    dict:
        ok        bool     是否成功定位到表盘（False 时 circle 为 None）
        corrected bool     True=真做了椭圆->圆白化；False=正圆(Hough)路径，仅平移
        circle    ndarray  校正图，与输入同分辨率，**圆心固定在图像正中心**，
                           圆外用常量 (30,30,30) 填充
        Minv      (2,3)    dst(校正图) -> src(输入图) 的仿射矩阵。
                           模型在 circle 上预测的关键点必须用它反算回输入图坐标，
                           否则点会整体错位（这是接入时最容易漏的一步）。
        M         (2,3)    src -> dst 前向矩阵（warpAffine 用的就是它）
        cx, cy    float    校正图坐标系下的圆心（恒为 w/2, h/2）
        r         float    校正后的圆半径（= 椭圆长短半轴均值）
        ratio     float|None  椭圆短轴/长轴（1.0 = 本来就是正圆）；Hough 路径为 None
        counts    dict     三色圆周扫描计数 R/G/Y
        ms        float    耗时毫秒
        reason    str|None 失败原因（ok=False 时才有）

    调用方注意：**ok=False 要降级用原图跑**，不能因为校正失败就整张丢掉结果。
    """
    h0, w0 = img_src.shape[:2]
    t0 = time.perf_counter()

    def _ms():
        return (time.perf_counter() - t0) * 1000.0

    work, scale = preprocess(img_src)          # 在 600 高工作图上检测（专利参数前提）
    dial = find_dial(work)
    if dial is None:
        return {"ok": False, "corrected": False, "circle": None, "Minv": None,
                "M": None, "cx": None, "cy": None, "r": None, "ratio": None,
                "counts": None, "ms": _ms(), "scale": float(scale),
                "ellipse": None, "d_center": None,
                "reason": "未找到表盘（椭圆/Hough 圆 + 红黄绿颜色验证均未通过）"}

    # 检测发生在 600 高工作图上；把参数按 1/scale 映射回输入图分辨率后再校正，
    # 这样输出与输入同分辨率，上游的坐标换算不用改。
    inv = 1.0 / scale
    target_c = np.array([w0 / 2.0, h0 / 2.0], dtype=np.float64)
    ratio = None
    ellipse = None

    # ---- 先取「检测到的圆心」，做可信度校验，再构造变换 ----
    if dial["corrected"]:
        cx_det, cy_det = float(dial["ellipse"]["cx"]) * inv, \
                         float(dial["ellipse"]["cy"]) * inv
    else:
        cx_det, cy_det = float(dial["cx"]) * inv, float(dial["cy"]) * inv
    d_center = math.hypot(cx_det - target_c[0], cy_det - target_c[1]) \
        / float(min(h0, w0))
    if d_center > Params.max_offcenter_ratio:
        return {"ok": False, "corrected": False, "circle": None, "Minv": None,
                "M": None, "cx": None, "cy": None, "r": None, "ratio": None,
                "counts": dial["counts"], "ms": _ms(), "scale": float(scale),
                "ellipse": None, "d_center": float(d_center),
                "reason": f"检测到的圆心偏离裁切图中心 {d_center:.2f}"
                          f"（> {Params.max_offcenter_ratio}），"
                          f"判定为找错目标，退回未校正的直裁图"}

    if dial["corrected"]:
        e = dict(dial["ellipse"])
        for k in ("cx", "cy", "ax1", "ax2", "r_out"):
            e[k] *= inv
        M = _ellipse_to_circle_matrix(e)       # src -> dst（白化 + 缩放到 r_out）
        # 白化把椭圆中心映回它自己，再叠平移把圆心摆到输出图正中心
        M[:, 2] += target_c - np.array([e["cx"], e["cy"]])
        r = float(e["r_out"])
        ratio = float(e["ratio"])
        ellipse = e
    else:
        # Hough 正圆路径：不做白化，只平移让圆心居中
        cx, cy = dial["cx"] * inv, dial["cy"] * inv
        M = np.array([[1.0, 0.0, target_c[0] - cx],
                      [0.0, 1.0, target_c[1] - cy]], dtype=np.float64)
        r = float(dial["r"] * inv)

    circle = cv2.warpAffine(img_src, M, (w0, h0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=(30, 30, 30))
    Minv = cv2.invertAffineTransform(M)
    return {"ok": True, "corrected": bool(dial["corrected"]), "circle": circle,
            "Minv": Minv, "M": M,
            "cx": float(target_c[0]), "cy": float(target_c[1]), "r": r,
            "ratio": ratio, "counts": dial["counts"], "ms": _ms(),
            "scale": float(scale), "reason": None, "ellipse": ellipse,
            "d_center": float(d_center)}


def map_points(M, pts):
    """用 2x3 仿射矩阵变换点集。pts: (N,2) -> (N,2)。"""
    P = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.transform(P, np.asarray(M, dtype=np.float32)).reshape(-1, 2)


# ----------------------------- 主流程 -----------------------------
def process(img_path):
    img_src = cv2.imread(img_path)
    if img_src is None:
        print("无法读取图片:", img_path)
        return False
    stem = img_path.rsplit(".", 1)[0]
    out_detect = stem + "_ellipse_result.jpg"
    out_circle = stem + "_circle.jpg"
    h0, w0 = img_src.shape[:2]
    print(f"输入: {img_path} ({w0}x{h0})")

    # S1 + S2 全在 correct_to_circle 里（在线流水线走的是同一个函数）
    res = correct_to_circle(img_src)
    print(f"S1 预处理: 等比缩放 x{res['scale']:.3f} + 3x3高斯")

    if not res["ok"]:
        print(res["reason"])
        return False

    counts = res["counts"]
    print(f"表盘(输入图坐标): 中心=({res['cx']:.1f}, {res['cy']:.1f}) "
          f"r={res['r']:.1f} 三色计数 R={counts[COLOR_RED]} "
          f"G={counts[COLOR_GREEN]} Y={counts[COLOR_YELLOW]}")
    print(f"耗时 {res['ms']:.0f}ms")

    # 标注图：原图 + 拟合到的椭圆（或圆）
    vis = img_src.copy()
    if res["corrected"]:
        e = res["ellipse"]
        print(f"椭圆->圆校正: ax1={e['ax1']:.1f} ax2={e['ax2']:.1f} 轴比={e['ratio']:.3f} "
              f"ang={e['ang']:.1f}° 边缘覆盖率={e['edge_cover']:.2f} 轮廓支撑点={e['support']}")
        cv2.ellipse(vis, (int(round(e["cx"])), int(round(e["cy"]))),
                    (max(1, int(round(e["ax1"]))), max(1, int(round(e["ax2"])))),
                    e["ang"], 0, 360, (255, 0, 0), 3)
        cv2.putText(vis, f"ellipse ratio={e['ratio']:.2f} ang={e['ang']:.0f} "
                         f"cover={e['edge_cover']:.2f}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    else:
        print("正圆路径（Hough 圆兜底，未做校正，仅平移居中）")
        cv2.circle(vis, (int(round(res["cx"])), int(round(res["cy"]))),
                   int(round(res["r"])), (255, 0, 0), 3)
        cv2.putText(vis, f"circle r={res['r']:.0f}", (15, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    cv2.imwrite(out_detect, vis)
    print("标注图:", out_detect)

    # 校正图（纯表盘，不画参考圆/圆心/十字线，避免干扰后续刻度识别）
    cv2.imwrite(out_circle, res["circle"])
    print(f"正圆表盘: {out_circle} ({w0}x{h0})")
    return True


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else IMG_PATH
    if not img_path:
        print("用法: python dial_ellipse_to_circle.py <图片路径>", file=sys.stderr)
        print("      （本精简包不含默认示例图，必须显式传路径）", file=sys.stderr)
        sys.exit(2)
    ok = process(img_path)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
