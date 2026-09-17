# -*- coding: utf-8 -*-
"""YOLO-pose 压力表**6 关键点**几何推理（双模型：针尖 + 几何）。

【本包性质】test-device-ocr = 主工程 device_ocr 的推理端精简包，
只保留网页跑起来必需的文件（清单见包根 README.md）；下文提到的训练脚本、
数据集目录（tools/、datasets/ 等）**不随本包提供**，仅作设计依据留档。

2026-09-17 口径（双模型 + 压力状态判定）
----------------------------------------
本模块负责「预测 6 个关键点」**并据此判出压力状态**（欠压/正常/超压）：
把圆心到 5 个非圆心点的射线都画出来，看「圆心 -> 针尖」这一条被夹在哪
两条相邻的边界射线之间 —— 这就是状态。判不了时明确说判不了，不硬猜。
判据实现见 state_from_kpts，触发开关见 classify(want_state=)。

==========================================================================
为什么从「1 个 6 点模型」改成「2 个模型」
==========================================================================
单个 6 点模型实测失败：关键点 loss 取 6 点平均（KeypointLoss 最后 .mean()），
针尖偏 300px 与偏 3px 对总 loss 的贡献都只有 1/6，梯度被 5 个「好学的」
静止点淹没 -> 模型放弃针尖，输出与表盘绑定的固定角（实测 r≈0.049）。

拆开成：
  针尖模型 (tip)  kpt_shape=[1,3]  -> 只出 1 个点：指针尖端
                  （当前权重 yolo-v8n-pose-best.pt，架构 yolov8n-pose）
  几何模型 (geom) kpt_shape=[5,3]  -> 出 5 个点：圆心 + 4 条色区边界
                  （当前权重 best_geom.pt，架构 yolov8m-pose）
两侧 loss 都 100% 作用在自己的点上，没有互相稀释。
两个模型的点数/架构互不绑定 —— 谁换权重都不需要改代码，路径在下面的
DEFAULT_WEIGHTS_TIP / DEFAULT_WEIGHTS_GEOM。

==========================================================================
统一输出：合并回 6 点（硬约定不变）
==========================================================================
    合并后 index  名字                来源
    ------------------------------------------------------------------
    kpt0          表盘圆心            geom[0]
    kpt1          欠压边界(红区起点)  geom[1]
    kpt2          欠压/正常分界       geom[2]
    kpt3          正常/超压分界       geom[3]
    kpt4          超压边界(黄区终点)  geom[4]
    kpt5          指针尖端            tip[0]
    ------------------------------------------------------------------
硬约定：**合并后 kpt0 = 圆心，最后一个 = 针尖**。下游（webapp_v2 /
state_from_kpts）依赖头尾两槽，换点位方案时不要动。

==========================================================================
向后兼容：单模型照跑
==========================================================================
`classify(bgr, model)` 老签名仍然可用 —— 只给一个 model 时按它的
kpt_shape 自己判定是「6 点单模型 / 2 点 / 1 点」，行为与改版前一致。
双模型走 `classify(bgr, model_geom=..., model_tip=...)`。

权重兼容（单模型路径）
  - kpt_shape = [1,3]   单关键点 = 指针尖端
  - kpt_shape = [2,3]   kpt0 = 表盘圆心, kpt1 = 指针尖端
  - kpt_shape = [5,3]   几何模型（5 点）：圆心 + 4 边界  <- 配合 [1,3] 使用
  - kpt_shape = [6,3]   kpt0 圆心 / kpt1 欠压边界(红区起点) / kpt2 欠压·正常分界
                        / kpt3 正常·超压分界 / kpt4 超压边界(黄区终点) / kpt5 针尖
                        <- 旧的单模型生产权重，仍可直接跑
以上 .pt 都不改代码直接跑，能力按关键点数自动升降。

输入：BGR 裁切图（按 YOLO 检测框裁好的方图）
输出 dict:
    ok        针尖置信度达标
    n_kpt     合并后关键点个数（双模型 = 6）
    n_kpt_tip / n_kpt_geom   各模型自己的点数（双模型时才有意义）
    kpts      [{name, x, y, conf}, ...]  合并后的关键点（裁切图坐标）<- 网页用这个
    tip       (x, y)  裁切图坐标系下的针尖（= kpts[-1] 的便利写法）
    tip_conf  float
    cx/cy/kpt0_conf   圆心（来自 geom[0]；单模型时取 kpts[0]）
    angle     圆心->针尖的角度（仅圆心置信度达标时有值）
    split     该结果是单模型还是双模型出的："single" / "tip+geom"
    reason    没找到针尖时的原因
    status    压力状态：'欠压'/'正常'/'超压'/'色带外'，判不了则为 '?'
    state     同上（None = 判不了，配合 state_reason 看原因）
    state_reason      判不了的原因；给出结论时为 None
    state_span        夹住针尖的那两条射线：
                      {lo, hi, lo_name, hi_name, lo_deg, hi_deg, width_deg}
    state_margin_deg  针尖离最近那条边界射线几度（小 = 结论不稳）
    boundary_angles   6 个点绕圆心的角度
    （以上 state_* 仅在 want_state=True 时有值）

用法（独立跑）：
  python classify_pose.py -i x.jpg                                  # 默认双模型
      （针尖 yolo-v8n-pose-best.pt + 几何 best_geom.pt -> 合并 6 点）
  python classify_pose.py -i x.jpg -w <单模型.pt>                    # 单模型（旧 6 点合一）
  python classify_pose.py -i x.jpg --weights-tip <针尖.pt> \
                                   -w2 <几何.pt>                    # 显式指定权重
  python classify_pose.py -i ../labeling/box_crops_v2 --debug-dir C:/tmp/pose_dbg
  python classify_pose.py -i x.jpg                    # 默认就打印压力状态
  python classify_pose.py -i x.jpg --no-state         # 只要 6 个点
  python classify_pose.py -i dir --state-csv s.csv    # 批量导出状态明细
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 默认权重
#   双模型（当前生产口径）：yolo-v8n-pose-best.pt（针尖 1 点，yolov8n-pose）
#                         + best_geom.pt        （几何 5 点，yolov8m-pose）
#   单模型（旧口径，向后兼容）：best.pt（6 点合一）
# 两个文件都存在时优先走双模型（见 resolve_default_weights）。
#
# 2026-09-17 换过的针尖权重：best_tip.pt（yolov8m-pose）-> yolo-v8n-pose-best.pt
#   （yolov8n-pose，用户另训）。两者 kpt_shape 都是 [1,3]、训练 imgsz 都是 960，
#   所以合并逻辑、分辨率、点名都不用动 —— **只有这一个路径常量变**。
#   旧的 m 版权重仍在磁盘上，要对比/回退就显式 `--weights-tip <path>`。
# ---------------------------------------------------------------------------
DEFAULT_WEIGHTS = ROOT / "best.pt"
DEFAULT_WEIGHTS_TIP = ROOT / "yolo-v8n-pose-best.pt"
DEFAULT_WEIGHTS_GEOM = ROOT / "best_geom.pt"

# 第二步 pose 的推理分辨率默认值。
# 本项目的 ckpt 是 imgsz=960 训出来的，而 ultralytics 推理默认 640 —— 不显式
# 指定就会「训练/推理分辨率不一致」，实测针尖误差明显变大、还会丢检测。
# 换权重时按该权重的训练 imgsz 改这里（或从命令行 --imgsz 传）。
DEFAULT_IMGSZ = 960

# pose 推理时的**框置信度**下限。刻意压到极低，不等于「针尖置信度阈值」。
# 为什么必须压：ultralytics 预测默认 conf=0.25，而第二步的输入已经是第一步
# 判定的表盘裁切图，图里一定有表盘 —— 这里的框只是关键点的载体，框定位置准
# 不准都不影响下游（我们只取关键点）。
# 实测代价：img/27.png 的 pose 框 conf=0.0988 < 0.25 被丢掉 -> 0 个结果 ->
# classify() 报 "no keypoints detected"，一个明明能出点的图被判成未检出。
# 真正的把关放在 tip 的关键点置信度上（classify 的 conf_thres）。
PRED_BOX_CONF = 0.01

CLASSES = ["欠压", "正常", "超压"]

# 关键点个数 -> 名字。最后一个是针尖，第 0 个（若有）是圆心。
# 5 点方案的顺序 = 几何模型输出顺序（= 6 点方案去掉针尖后的前 5 槽）：
#   换点位方案时不要动顺序，classify() 合并双模型时按 index 直接拼。
KPT_NAMES = {
    1: ["指针尖端"],
    2: ["表盘圆心", "指针尖端"],
    5: ["表盘圆心", "欠压边界", "欠压/正常分界", "正常/超压分界", "超压边界"],
    6: ["表盘圆心", "欠压边界", "欠压/正常分界", "正常/超压分界", "超压边界",
        "指针尖端"],
}

# ---------------------------------------------------------------------------
# 双模型任务名（与 classifier/train_pose.py 的 TASKS 键保持一致）
#   tip  = 最后 1 个点（针尖）
#   geom = 前 5 个点（圆心 + 4 边界）
# classify() 按 kpt_shape 自动判角色，不依赖文件名；这两个常量只用于
# 「给用户看的分组名」和 CLI 默认值。
# ---------------------------------------------------------------------------
TASK_TIP = "tip"
TASK_GEOM = "geom"


def kpt_names(n: int):
    """按关键点个数给出名字列表，个数不认识时兜底。"""
    if n in KPT_NAMES:
        return list(KPT_NAMES[n])
    return [f"kpt{i}" for i in range(max(0, n - 1))] + ["指针尖端"]


def resolve_default_weights():
    """决定默认走双模型还是单模型。

    返回 (tip_path | None, geom_path | None, single_path | None)：
      - 两个专用权重都在 -> 双模型（tip, geom, None）
      - 否则若有 best.pt -> 单模型（None, None, best.pt）
      - 再否则：谁在就用谁（例如只训了针尖模型）
    webapp_v2/server.py 也用这个函数，保证「本机/服务器上默认用什么」只有一处定义。
    """
    tip, geom = DEFAULT_WEIGHTS_TIP, DEFAULT_WEIGHTS_GEOM
    if tip.is_file() and geom.is_file():
        return tip, geom, None
    if DEFAULT_WEIGHTS.is_file():
        return None, None, DEFAULT_WEIGHTS
    if tip.is_file():
        return tip, None, None
    if geom.is_file():
        return None, geom, None
    return None, None, None


# 默认角度规则（与 classify_traditional 保持一致）
DEFAULT_RANGES = {
    "欠压": (200.0, 280.0),
    "正常": (280.0, 340.0),
    "超压": (340.0, 200.0),     # wrap-around
}
DEFAULT_SHIFT_DEG = 0.0


# ----------------------------- 单图推理
def kpts_to_angle(cx, cy, tx, ty, shift=0.0):
    """(cx, cy) -> (tx, ty) 在图像坐标下的角度（0..360）。"""
    a = math.degrees(math.atan2(ty - cy, tx - cx))
    if a < 0:
        a += 360.0
    return (a + shift) % 360.0


def angle_to_status(angle, ranges=None, shift=0.0):
    if ranges is None:
        ranges = DEFAULT_RANGES
    a = (angle + shift) % 360.0
    for k, (lo, hi) in ranges.items():
        if lo <= hi:
            if lo <= a < hi:
                return k
        else:
            if a >= lo or a < hi:
                return k
    return "?"


# --------------------- 6 点方案：由边界点判三态（离线工具，不在生产链路上）
# 与标注工具 annotate_interactive.geometry_report() 的口径逐字一致 ——
# 那边是标注时给人当场核对用的，这边留着做离线比对（例如核「模型预测的
# 分界线 vs 人工标的分界线差多少」）。**网页端不用它**：classify() 默认
# 不算三态，要算得显式 want_state=True。
STATE_OUTSIDE = "色带外"        # 针尖落在白盘面上（表盘正常区之外的空白刻度区）
_ARC_SEGS = ("欠压", "正常", "超压")
# 每个状态区间由「哪两条射线」夹出来（数字 = 6 点里的槽位）。
# 与 _ARC_SEGS 一一对应 —— 换点位方案时这两个元组必须一起动：
#   欠压 = [ kpt1 欠压边界      -> kpt2 欠压/正常分界 ]
#   正常 = [ kpt2 欠压/正常分界 -> kpt3 正常/超压分界 ]
#   超压 = [ kpt3 正常/超压分界 -> kpt4 超压边界    ]
_ARC_HULLS = ((1, 2), (2, 3), (3, 4))

# 状态 -> 叠加图上显示用的 ASCII 名与 BGR 颜色。
# 为什么是 ASCII：cv2.putText 画不了中文（会变成一串 ???），图上只能写英文，
# 中文状态名交给网页 HTML 显示。颜色刻意跟表盘色区对齐：欠压红 / 正常绿 /
# 超压黄，看图时不用再对图例。
STATE_ASCII = {"欠压": "UNDER", "正常": "NORMAL", "超压": "OVER",
               STATE_OUTSIDE: "OUTSIDE"}
STATE_BGR = {"欠压": (60, 60, 235), "正常": (70, 190, 70),
             "超压": (0, 180, 250), STATE_OUTSIDE: (205, 205, 205)}


def _ang_deg(c, p):
    """圆心 c -> 点 p 的角度 (0..360)。图像坐标 y 向下，角度增大即顺时针。"""
    a = math.degrees(math.atan2(p[1] - c[1], p[0] - c[0]))
    return a + 360.0 if a < 0 else a


def _arc_contains(a, lo, hi):
    """a 是否落在 lo->hi 这段圆弧上（lo > hi 表示跨 0°）。"""
    if lo <= hi:
        return lo <= a < hi
    return a >= lo or a < hi


def _detail(reason, angles=None, order=None, span=None, margin=None,
            tip_deg=None):
    """统一构造 state_from_kpts 的 detail —— 键永远齐全（值为 None 表示不适用）。

    键固定是为了下游好写：调用方直接读 detail["span"]、detail["margin_deg"]，
    不用到处 .get() 兜底。加字段时只改这一个地方。
    """
    return {"reason": reason, "angles": angles, "order": order,
            "span": span, "margin_deg": margin, "tip_deg": tip_deg}


def state_from_kpts(kpts, conf_thres=0.20):
    """6 点关键点 -> (state, detail)。**压力表状态判定的唯一实现。**

    判据（角度一律以**圆心 kpt0** 为极点）—— 说白了就是把「圆心->各点」的
    射线都画出来，看**圆心->针尖**这条射线被夹在哪两条相邻的边界射线之间：

        [ang(kpt1), ang(kpt2))  欠压    夹在 欠压边界 · 欠压/正常分界 之间
        [ang(kpt2), ang(kpt3))  正常    夹在 欠压/正常分界 · 正常/超压分界 之间
        [ang(kpt3), ang(kpt4))  超压    夹在 正常/超压分界 · 超压边界 之间
        其余                    色带外  针尖落在色带两端之外的白盘面

    判据用的是**射线方向（角度）**，不是「针尖离圆心多远」—— 所以指针画多长
    都不影响结论；真正影响结论的只有圆心位置（它是所有角度的极点）。

    两道前置校验，任一不过就不给结论 —— 宁可说不确定，也不给错的三态：
      1. 圆心 + 4 个边界点 + 针尖 的置信度都要 >= conf_thres；
         （边界点学得比针尖差，低置信度的边界点会把分界线拉歪，
           而分界线错一点，判出来的状态就整体偏一档）
      2. 4 个边界点的角度在绕一圈的次序上必须单调递增
         （以 kpt1 为 0 起算：d2 < d3 < d4）。
         顺序乱 = 模型把跨色区的点预测串了（例如把「正常/超压分界」预测到了
         「欠压/正常分界」之前），这时任何圆弧判据都会给出荒唐结论。

    Parameters
    ----------
    kpts : (K,3) ndarray  (x, y, conf)，图像坐标
    conf_thres : float    生产用默认 0.20；离线核 GT 标点时传 0 关掉置信度门

    Returns
    -------
    (state, detail)
      state  : '欠压' / '正常' / '超压' / '色带外' / None(判不了)
      detail : {reason, angles, order, span, margin_deg, tip_deg}
        reason     None = 给出了结论；否则是「为什么判不了」的原话
        angles     6 个点各自绕圆心的角度（度），下标 = 点位顺序  <- 排查用
        order      4 个边界点相对 kpt1 的有向角度，递增才说明次序没串
        span       夹住针尖的那两条射线：
                   {lo, hi, lo_name, hi_name, lo_deg, hi_deg, width_deg}
                   lo/hi 是 6 点里的槽位号，lo_name/hi_name 是中文点名
        margin_deg 针尖离**最近**那条边界射线还差几度（0 = 正压在分界线上）
                   <- 数很小就说明结论在边界附近，模型抖一抖就会跳档，别太当真
        tip_deg    圆心 -> 针尖 的角度
    """
    kpts = np.asarray(kpts, dtype=np.float32)
    K = int(kpts.shape[0])
    if K < 6:
        return None, _detail(f"关键点数 {K} < 6，无法用边界点判三态")

    cen = (float(kpts[0][0]), float(kpts[0][1]))
    confs = [float(kpts[i][2]) for i in range(6)]
    if min(confs) < conf_thres:
        low = [f"{kpt_names(6)[i]} {confs[i]:.2f}" for i in range(6)
               if confs[i] < conf_thres]
        return None, _detail(f"关键点置信度不足（{', '.join(low)}）")

    ang = [_ang_deg(cen, (float(kpts[i][0]), float(kpts[i][1])))
           for i in range(6)]
    b = ang[1:5]                       # 4 个边界点角度
    # 以 kpt1 为 0 起算的有向角度：d2/d3/d4 必须递增，才说明边界次序没串
    d = [(x - b[0]) % 360.0 for x in b]
    det = _detail(None, angles=[round(x, 2) for x in ang],
                  order=[round(x, 2) for x in d], tip_deg=round(ang[5], 2))
    if not (d[1] < d[2] < d[3]):
        det["reason"] = (f"边界点角度次序异常（相对首个边界 {det['order']}），"
                         f"模型把色区边界预测串了")
        return None, det

    tip_a = ang[5]
    names = kpt_names(6)
    for i, nm in enumerate(_ARC_SEGS):
        if _arc_contains(tip_a, b[i], b[i + 1]):
            lo, hi = _ARC_HULLS[i]
            det["span"] = {"lo": lo, "hi": hi,
                           "lo_name": names[lo], "hi_name": names[hi],
                           "lo_deg": round(b[i], 2),
                           "hi_deg": round(b[i + 1], 2),
                           "width_deg": round((b[i + 1] - b[i]) % 360.0, 2)}
            det["margin_deg"] = round(min((tip_a - b[i]) % 360.0,
                                          (b[i + 1] - tip_a) % 360.0), 2)
            return nm, det
    # 不在任何色区里 = 落在白盘面（色带两端之外）
    det["reason"] = "针尖不在色带范围内（落在白盘面）"
    return STATE_OUTSIDE, det


def parse_best_kpts(res):
    """从 ultralytics Results 里取出「置信度最高实例」的关键点。

    返回 (kpts, reason)：kpts 是 (K, 3) 的 float32 ndarray (x, y, conf)；
    取不到时 kpts=None 并给出 reason。

    ⚠ ultralytics 8.4.x 的 res.keypoints 是 Results.Keypoints 对象，
      对它做索引 kp[best] 不会切出第 best 个实例（返回的还是整个对象），
      旧写法 np.asarray(kp[best]).reshape(-1,3) 会直接抛
      "setting an array element with a sequence"。
      唯一可靠的取法：kp.data -> (n_inst, K, 3) 的 tensor (x, y, conf)。
    """
    kp_all = getattr(res.keypoints, "data", None)
    if kp_all is None:
        kp_all = res.keypoints
    if kp_all is None or len(kp_all) == 0:
        return None, "no keypoints detected"

    # 取置信度最高的那一个实例
    if getattr(res.boxes, "conf", None) is not None:
        best = int(np.argmax(res.boxes.conf.cpu().numpy()))
    else:
        best = 0

    if hasattr(kp_all, "cpu"):
        kp_all = kp_all.cpu().numpy()
    kp_all = np.asarray(kp_all, dtype=np.float32)

    # 统一成 (n_inst, K, 3)
    if kp_all.ndim == 1:
        kp_all = kp_all.reshape(1, -1, 3)
    elif kp_all.ndim == 2:
        kp_all = kp_all.reshape(kp_all.shape[0], -1, 3)
    if kp_all.ndim != 3 or kp_all.shape[-1] != 3:
        return None, f"keypoint shape 异常: {kp_all.shape}"

    if not (0 <= best < kp_all.shape[0]):
        best = 0
    return kp_all[best], None


def kpt_shape_of(model):
    """读已加载 pose 模型的关键点形状，返回 [n, 3]；读不到返回 []。

    为什么要多处兜底（实测踩过）：
      - `model.model.kpt_shape` 只有走 ultralytics **训练/保存**管线的 ckpt 才
        会有这个属性。用 PoseModel(...) 直接 torch.save 出来的 ckpt 没有它，
        但模型本身是好的（能正常推理）。
      - Pose head 上的 `model.model[-1].kpt_shape` 是**始终**存在的（它是 head
        的成员变量），所以它才是最稳的来源。
      - 再退一步读 inner.yaml['kpt_shape']。
    只读第一处会在加载非训练产物时静默拿到 0 点，进而 kpt_names 兜底成
    ["指针尖端"]，页面显示成一团乱码般的错名字。
    """
    inner = getattr(model, "model", None)
    if inner is None:
        return []
    # 1) PoseModel.kpt_shape（训练产物有）
    ks = getattr(inner, "kpt_shape", None)
    if ks:
        return list(ks)
    # 2) Pose head 的 kpt_shape（始终有）
    try:
        head = inner.model[-1]
    except Exception:
        head = None
    ks = getattr(head, "kpt_shape", None)
    if ks:
        return list(ks)
    # 3) inner.yaml
    try:
        yml = getattr(inner, "yaml", None) or {}
        if isinstance(yml, dict) and yml.get("kpt_shape"):
            return list(yml["kpt_shape"])
    except Exception:
        pass
    return []


def model_n_kpt(model) -> int:
    """读已加载 pose 模型的关键点个数（读不到返回 0）。"""
    ks = kpt_shape_of(model)
    try:
        return int(ks[0]) if ks else 0
    except Exception:
        return 0


def _infer_kpts(bgr, model, imgsz, box_conf):
    """对一个 pose 模型跑一次推理，返回 (kpts, reason)。

    kpts 是 (K,3) float32 ndarray；失败时 kpts=None 且 reason 是原因。
    抽出成函数是因为双模型要跑两遍，两遍的推理参数完全一样。
    """
    # 推理：ultralytics 输入 RGB；这里直接给 BGR 它会自己转
    kw = {"verbose": False}
    if imgsz:
        kw["imgsz"] = int(imgsz)
    if box_conf is not None:
        kw["conf"] = float(box_conf)
    res = model(bgr, **kw)[0]
    return parse_best_kpts(res)


def _merge_two(k_tip, k_geom, conf_thres):
    """把「针尖 1 点」与「几何 5 点」拼回统一的 6 点 (K,3) ndarray。

    拼接顺序（硬约定，不要改）：
        geom[0..4] 放前面（圆心理所当然占 kpt0），tip[0] 放最后（针尖）。
    这样合并结果与旧的 6 点单模型**逐槽对齐** —— 下游
    （webapp_v2 的 6 行点名、state_from_kpts 的 kpts[0]/kpts[5]）一行都不用改。

    某个模型没出点时，对应槽位用 (0, 0, 0) 占位：conf=0 会让
    state_from_kpts 的置信度校验直接拦下，不会给出错结论。
    """
    rows = []
    if k_geom is None:
        rows.extend([[0.0, 0.0, 0.0]] * 5)
    else:
        g = np.asarray(k_geom, dtype=np.float32)
        if g.shape[0] < 5:
            # geom 模型点数不对（权重放错了/换成别的版本）：补零并保留已有的
            g = np.vstack([g, np.zeros((5 - g.shape[0], 3), dtype=np.float32)])
        rows.extend(g[:5].tolist())
    if k_tip is None:
        rows.append([0.0, 0.0, 0.0])
    else:
        rows.append(np.asarray(k_tip, dtype=np.float32)[-1].tolist())
    return np.asarray(rows, dtype=np.float32)


def _kpts_dict(kpts):
    """(K,3) ndarray -> [{name, x, y, conf}, ...]（裁切图坐标，已 round）。"""
    K = int(kpts.shape[0])
    names = kpt_names(K)
    return [{"name": names[i],
             "x": round(float(kpts[i][0]), 2),
             "y": round(float(kpts[i][1]), 2),
             "conf": round(float(kpts[i][2]), 4)} for i in range(K)]


def classify(bgr, model=None, ranges=None, shift=DEFAULT_SHIFT_DEG,
             conf_thres=0.20, draw_overlay=None, draw_center=False,
             imgsz=DEFAULT_IMGSZ, box_conf=PRED_BOX_CONF, want_state=True,
             model_tip=None, model_geom=None):
    """一步到位：BGR 裁切图 -> 6 个关键点 + 针尖 dict。

    两种调用姿势：

      双模型（当前生产口径）
        classify(bgr, model_geom=geom, model_tip=tip)
        -> 两个模型各跑一次，合并回 6 点（kpt0=圆心 ... kpt5=针尖）

      单模型（向后兼容，行为与改版前完全一致）
        classify(bgr, model)
        -> 按该模型的 kpt_shape 自己判 1/2/5/6 点

    Parameters
    ----------
    bgr : np.ndarray   BGR 裁切图
    model : ultralytics.YOLO | None   单模型权重（双模型时不用传）
    model_tip : ultralytics.YOLO | None   针尖模型（1 点）
    model_geom : ultralytics.YOLO | None  几何模型（5 点）
    ranges : dict | None   角度 -> 状态（仅 2 关键点权重下会用到）
    shift : float   角度整体平移
    conf_thres : float  **关键点**置信度阈值（真正把关的那个）
    imgsz : int | None  推理分辨率。默认 960 = 本项目权重的训练分辨率；
                        传 None 用 ultralytics 自己的默认（640，一般不要）
    box_conf : float  推理时的**框**置信度下限，默认 0.01（见 PRED_BOX_CONF）。
                        别用 ultralytics 默认的 0.25，那会把低对比裁切图整张丢掉
    draw_overlay : np.ndarray | None  给了就把关键点画到这张 BGR 图上
    draw_center : bool  是否顺便把圆心也画上（默认 False，圆心不重要）
    want_state : bool  是否算压力三态（欠压/正常/超压）。**默认 True** ——
                        6 点齐全时交给 state_from_kpts 判；判不了就给
                        state=None + state_reason 说明原因（绝不硬猜）。
                        只要纯几何量、不要状态结论时传 False。
                        只有 2 点 / 5 点权重时天然算不了，state 恒为 None。

    Returns
    -------
    dict: ok, tip, tip_conf, n_kpt, n_kpt_tip, n_kpt_geom, kpts, split,
          cx, cy, kpt0_conf, angle, reason, status
          + want_state 时: state / state_reason / state_span /
            state_margin_deg / boundary_angles
    """
    out = {"ok": False, "status": "?", "angle": None, "n_kpt": None,
           "n_kpt_tip": None, "n_kpt_geom": None, "split": None,
           "tip": None, "tip_conf": None, "kpts": None,
           "cx": None, "cy": None, "kpt0_conf": None,
           "state": None, "state_reason": None, "boundary_angles": None,
           "state_span": None, "state_margin_deg": None, "state_detail": None,
           "reason": None}

    if bgr is None or bgr.size == 0:
        out["reason"] = "empty image"
        return out

    dual = model_tip is not None or model_geom is not None
    kpts = None

    if dual:
        # ----------------- 双模型：各跑一次，合并回 6 点
        out["split"] = "tip+geom"
        k_tip = k_geom = None
        g_reason = t_reason = None
        if model_geom is not None:
            k_geom, g_reason = _infer_kpts(bgr, model_geom, imgsz, box_conf)
            out["n_kpt_geom"] = 0 if k_geom is None else int(k_geom.shape[0])
        else:
            out["n_kpt_geom"] = None
        if model_tip is not None:
            k_tip, t_reason = _infer_kpts(bgr, model_tip, imgsz, box_conf)
            out["n_kpt_tip"] = 0 if k_tip is None else int(k_tip.shape[0])
        else:
            out["n_kpt_tip"] = None

        if k_tip is None and k_geom is None:
            both = "; ".join(x for x in (
                f"针尖模型: {t_reason}" if model_tip is not None else "",
                f"几何模型: {g_reason}" if model_geom is not None else "",
            ) if x)
            out["reason"] = f"两个模型都没出关键点 ({both})"
            return _maybe_draw(out, bgr, draw_overlay, None, None,
                               draw_center, None)
        if k_tip is None:
            out["reason"] = f"针尖模型未出点: {t_reason}"
        elif k_geom is None:
            # 针尖有、几何没出：仍把针尖交出去（下层可只看针尖），
            # 但把缺失原因写进 reason，别让人以为几何点是 (0,0) 是真的。
            out["reason"] = f"几何模型未出点: {g_reason}"
        kpts = _merge_two(k_tip, k_geom, conf_thres)
    else:
        # ----------------- 单模型：老路径，行为不变
        out["split"] = "single"
        if model is None:
            out["reason"] = "没有传入任何 pose 模型"
            return out
        kpts, reason = _infer_kpts(bgr, model, imgsz, box_conf)
        if kpts is None:
            out["reason"] = reason
            return _maybe_draw(out, bgr, draw_overlay, None, None,
                               draw_center, None)
        out["n_kpt_tip"] = int(kpts.shape[0])

    K = int(kpts.shape[0])
    out["n_kpt"] = K

    # 全部关键点（裁切图坐标）—— 网页要按名字逐个显示/画出来，所以在这里一次给全
    out["kpts"] = _kpts_dict(kpts)

    # 统一约定：针尖 = 最后一个关键点；圆心 = 第 0 个（若 K>=2）
    tx, ty, c1 = (float(v) for v in kpts[K - 1])
    if K >= 2:
        cx, cy, c0 = (float(v) for v in kpts[0])
        out["cx"], out["cy"] = int(round(cx)), int(round(cy))
        out["kpt0_conf"] = c0
    else:
        cx = cy = c0 = None

    out["tip"] = (int(round(tx)), int(round(ty)))
    out["tip_conf"] = c1
    center = None if cx is None else (cx, cy)

    # 压力三态：**默认算**（2026-09-17 起）。判据 = 圆心->针尖 这条射线被
    # 哪两条相邻的边界射线夹住，见 state_from_kpts。
    # 判不了时 state 保持 None，原因写进 state_reason —— **绝不**退回
    # 「硬编码角度区间」（DEFAULT_ANGLE_RULES）去凑一个结论：那套区间只对
    # 特定安装姿态成立，与预测出来的边界点无关，混用会给出看着合理但完全
    # 错的状态，比"不知道"危险得多。
    if want_state:
        if K >= 6:
            st, st_det = state_from_kpts(kpts, conf_thres)
            out["state"], out["state_reason"] = st, st_det["reason"]
            out["boundary_angles"] = st_det["angles"]
            out["state_span"] = st_det["span"]
            out["state_margin_deg"] = st_det["margin_deg"]
            out["state_detail"] = st_det
        else:
            out["state_reason"] = f"关键点数 {K} < 6，该权重无边界点，判不了三态"

    if c1 < conf_thres:
        out["reason"] = (f"针尖置信度 {c1:.2f} < {conf_thres}"
                         + (f"（圆心 {c0:.2f}）" if c0 is not None else ""))
        return _maybe_draw(out, bgr, draw_overlay, (tx, ty), c1,
                           draw_center, center)

    # 角度只在「模型给了圆心、且圆心置信度达标」时才算 —— 针尖本身不需要它
    ang = status = None
    if cx is not None and c0 >= conf_thres:
        ang = kpts_to_angle(cx, cy, tx, ty, shift=0.0)
        if out["state"] is not None:
            status = out["state"]
        elif not want_state:
            # 显式关掉三态时，才退回「固定角度区间」的传统口径
            # （只对特定安装姿态的表盘成立，属遗留能力）
            status = angle_to_status(ang, ranges=ranges, shift=shift)
        # else: 要判但判不了（置信度不足 / 边界点次序乱）-> 就是"不知道"

    # 双模型下若几何模型没出点，reason 里留着那条提示（但 ok 仍为 True，
    # 因为针尖本身是可信的），调用方据此判断「6 点是否齐全」。
    out.update({"ok": True, "status": status or "?"})
    if not (out.get("reason") and "几何模型未出点" in str(out["reason"])):
        out["reason"] = None
    return _maybe_draw(out, bgr, draw_overlay, (tx, ty), c1, draw_center, center)


def _maybe_draw(out, bgr, draw_overlay, tip, tip_conf, draw_center=False,
                center=None):
    """只在传了 draw_overlay 时才画。默认只画针尖，圆心可选，6 点时补画边界点。

    ⚠ draw_overlay 是「画布」本身而不是开关：画在它上面，输入 bgr 保持干净。
      （旧版把它当开关、画在了 bgr 上，于是 `overlay = bgr.copy()` 那种调用
      拿到的永远是一张没画过的图 —— CLI --debug-dir 的输出一直是白图。）
    """
    if draw_overlay is None or bgr is None:
        return out
    img = draw_overlay
    H, W = img.shape[:2]
    s = max(H, W) / 960.0                    # 观感按 960 方图归一
    r = max(5, int(round(9 * s)))
    L = r * 2
    lw = max(1, int(round(2 * s)))
    fs = max(0.45, 0.60 * s)

    def _cross(p, color):
        x, y = int(round(p[0])), int(round(p[1]))
        for w, c in ((lw + 3, (0, 0, 0)), (lw + 1, color)):
            cv2.line(img, (x - L, y), (x + L, y), c, w, cv2.LINE_AA)
            cv2.line(img, (x, y - L), (x, y + L), c, w, cv2.LINE_AA)
        cv2.circle(img, (x, y), r, (0, 0, 0), lw, cv2.LINE_AA)
        return x, y

    def _label(txt, x, y, color):
        for w, c in ((lw + 3, (0, 0, 0)), (lw + 1, color)):
            cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fs, c, w, cv2.LINE_AA)

    # 6 点权重：把中间的边界点也画出来（编号 1~4）。编号是关键 —— 只画针尖的话，
    # 「边界点有没有互相串位」在缩略图上根本看不出来，而这正好是后续要评估的点。
    # 注：conf==0 的点是双模型合并时的占位（该模型没出点），跳过不画，
    #     否则会在图像左上角堆一串假的十字叉。
    klist = out.get("kpts") or []
    if len(klist) > 2:
        for i in range(1, len(klist) - 1):
            kp = klist[i]
            if not kp.get("conf"):
                continue
            x, y = _cross((kp["x"], kp["y"]), (255, 190, 0))
            _label(str(i), x + L + 6, y + L + 4, (255, 190, 0))

    # ---- 压力状态的可视化：把「圆心 -> 每个点」的射线都画出来 ----------------
    # 这不是装饰，是判据本身的可视化：判据 = 看「圆心->针尖」这条射线落在哪两条
    # 相邻的「圆心->边界点」射线之间。不画射线，判出来的状态就只能盲信。
    #   灰细线 = 4 条边界射线   亮粗线 = 针尖射线 + 被夹住的那段圆弧
    state_col = STATE_BGR.get(out.get("state"), (200, 200, 200))
    need_center = bool(draw_center) or out.get("state_span") is not None
    if need_center and center is not None:
        x, y = _cross(center, (80, 80, 255))
        _label("center", x + L + 6, y + L + 4, (80, 80, 255))

    span = out.get("state_span")
    if center is not None and tip is not None and len(klist) >= 6:
        cx0, cy0 = int(round(center[0])), int(round(center[1]))
        for i in range(1, 5):
            kp = klist[i]
            if not kp.get("conf"):
                continue
            cv2.line(img, (cx0, cy0),
                     (int(round(kp["x"])), int(round(kp["y"]))),
                     (160, 160, 160), max(1, lw - 1), cv2.LINE_AA)
        cv2.line(img, (cx0, cy0), (int(round(tip[0])), int(round(tip[1]))),
                 state_col, lw + 1, cv2.LINE_AA)
        if span is not None:
            # 被夹住的那段圆弧：自己按角度步进连折线画，避开 cv2.ellipse
            # 角度方向的约定歧义（我们的角度是 y 向下、顺时针为正）。
            r_arc = int(round(math.hypot(tip[0] - center[0],
                                         tip[1] - center[1])))
            n_seg = max(8, int(span["width_deg"] / 2.0))
            pts = []
            for j in range(n_seg + 1):
                a = math.radians(span["lo_deg"]
                                 + span["width_deg"] * j / n_seg)
                pts.append((int(round(cx0 + r_arc * math.cos(a))),
                            int(round(cy0 + r_arc * math.sin(a)))))
            cv2.polylines(img, [np.asarray(pts, dtype=np.int32)], False,
                          state_col, max(2, lw + 1), cv2.LINE_AA)
        txt = STATE_ASCII.get(out.get("state"))
        if txt:
            mg = out.get("state_margin_deg")
            txt += "" if mg is None else f"  margin {mg:.1f}deg"
            _label(txt, 8, int(30 * s) + 8, state_col)

    if tip is not None and tip_conf:
        x, y = _cross(tip, (0, 215, 255))
        txt = "tip" + ("" if tip_conf is None else f" {tip_conf:.2f}")
        _label(txt, x + L + 6, y - 8, (0, 215, 255))

    out["overlay_shape"] = (H, W)
    return out


# ----------------------------- CLI
def _load_model(weights_path: str):
    from ultralytics import YOLO
    if not os.path.exists(weights_path):
        raise SystemExit(
            f"找不到权重: {weights_path}\n"
            f"  先跑训练: python train_pose.py --task tip|geom\n")
    return YOLO(weights_path)


# 公开别名：webapp_v2/server.py 等外部模块用这个名字，别再用带下划线的私有名
load_model = _load_model


def arch_name(model, default="pose"):
    """从已加载的 pose 模型里读出「用的是哪个架构」，例 'yolov8m-pose.pt'。

    为什么读而不写死：模型名散落在 README / 前端标签 / 日志里就一定会漂移
    （本项目已踩过一次：代码换成 yolov8m 后文档还印着 yolov8n）。
    唯一可信来源是 ckpt 自己 —— train_args['model'] 是训练时传的权重路径。

    读不到就依次退到 model.yaml / 参数量，最后给 default。
    """
    ck = getattr(model, "ckpt", None) or {}
    ta = ck.get("train_args") or {}
    w = ta.get("model") or ta.get("weights") or ""
    if w:
        return os.path.basename(str(w))
    inner = getattr(model, "model", None)
    for attr in ("yaml_file", "yaml_path"):
        v = getattr(inner, attr, None)
        if v:
            return os.path.basename(str(v))
    try:
        yml = getattr(inner, "yaml", None) or {}
        if isinstance(yml, dict):
            for k in ("yaml_file", "yaml_path"):
                if yml.get(k):
                    return os.path.basename(str(yml[k]))
    except Exception:
        pass
    return default


def n_params(model):
    """参数量（M），用于在没有架构名时区分 n/s/m/l/x。"""
    try:
        return round(sum(p.numel() for p in model.model.parameters()) / 1e6, 2)
    except Exception:
        return None


def model_info(model, label="pose"):
    """打包一个模型的一句话描述，供 CLI / server 的 health 复用。

    返回 {task, arch, params_m, kpt_shape, n_kpt, label}
    kpt_shape 走 kpt_shape_of() 的多级兜底（见那里的注释）。
    """
    ks = kpt_shape_of(model)
    return {
        "label": label,
        "arch": arch_name(model),
        "params_m": n_params(model),
        "kpt_shape": ks,
        "n_kpt": int(ks[0]) if ks else 0,
    }


def _gather(path):
    if os.path.isfile(path):
        return [path]
    out = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        out.extend(Path(path).glob(ext))
    return sorted([str(p) for p in out])


def describe_state(r):
    """把 classify() 的结果翻成人能读的一句状态说明。

    返回 (short, long)：
      short  汇总计数用的短标签（'欠压'/'正常'/'超压'/'色带外'/'未判定'）
      long   逐图打印用的详述，讲清「夹在哪两条射线之间、离边界多远」

    单独抽出来是因为「针尖落在色带外」和「判不了」是两件事：前者是一个
    正经结论（指针确实指在刻度色带之外），后者是模型没给出可信的点。
    汇总时必须分开数，不然「判不了」会被当成一个状态类目。
    """
    st = r.get("state")
    if st is None:
        return "未判定", (r.get("state_reason") or "无结论")
    sp = r.get("state_span")
    if sp is None:                       # 色带外：没有夹住它的两条射线
        return st, f"{st}（针尖不在色带范围内，落在白盘面）"
    bits = [f"夹在 [{sp['lo_name']} {sp['lo_deg']}° "
            f"-> {sp['hi_name']} {sp['hi_deg']}°]"]
    td = r.get("state_detail") or {}
    if td.get("tip_deg") is not None:
        bits.append(f"针尖 {td['tip_deg']}°")
    if r.get("state_margin_deg") is not None:
        bits.append(f"离最近边界 {r['state_margin_deg']}°")
    return st, f"{st}  " + "  ".join(bits)


def main():
    ap = argparse.ArgumentParser(description="YOLO-pose 压力表 6 关键点推理（双模型）")
    ap.add_argument("-i", "--input", required=True,
                    help="单张图或目录")
    ap.add_argument("-w", "--weights", default=None,
                    help=f"单模型权重（6 点合一）。不给 -w2 时用它；"
                         f"默认 {DEFAULT_WEIGHTS}。")
    ap.add_argument("-w2", "--weights-geom", default=None,
                    help=f"几何模型权重（5 点：圆心+4边界）。给了它就与 --weights "
                         f"组成双模型；默认 {DEFAULT_WEIGHTS_GEOM}")
    ap.add_argument("--weights-tip", default=None,
                    help=f"针尖模型权重（1 点）。与 --weights-geom 配对使用；"
                         f"默认 {DEFAULT_WEIGHTS_TIP}")
    ap.add_argument("--conf", type=float, default=0.20,
                    help="针尖关键点置信度阈值")
    ap.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ,
                    help=f"推理分辨率，默认 {DEFAULT_IMGSZ}（= 训练分辨率）；"
                         f"传 0 用 ultralytics 默认 640")
    ap.add_argument("--shift", type=float, default=DEFAULT_SHIFT_DEG)
    ap.add_argument("--state", dest="state", action="store_true", default=True,
                    help="判定压力三态（欠压/正常/超压）。**默认开启**："
                         "6 点齐全就判，判不了会说明原因")
    ap.add_argument("--no-state", dest="state", action="store_false",
                    help="关掉三态判定，只输出 6 个点（纯几何量）")
    ap.add_argument("--state-csv", default=None,
                    help="把每张图的状态/区间/角度导出成 CSV（给批量核对用）")
    ap.add_argument("--draw-center", action="store_true",
                    help="调试用：把圆心也画出来（默认只画针尖）")
    ap.add_argument("--debug-dir", default=None,
                    help="画图输出目录（每张生成 <stem>.tip.jpg）")
    args = ap.parse_args()

    # ---- 决定用哪套权重 ----------------------------------------------------
    # 显式给了 --weights-geom / --weights-tip -> 双模型；
    # 只给 --weights -> 单模型（严格按用户说的来，不偷偷加另一个）。
    # 都不给 -> 用 resolve_default_weights() 的默认口径。
    geo_arg = args.weights_geom
    tip_arg = args.weights_tip
    dual_wanted = bool(geo_arg or tip_arg)

    if dual_wanted:
        w_geom = Path(geo_arg) if geo_arg else DEFAULT_WEIGHTS_GEOM
        w_tip = Path(tip_arg) if tip_arg else DEFAULT_WEIGHTS_TIP
        if not w_geom.is_file() and not w_tip.is_file():
            raise SystemExit(
                f"双模型的权重都不存在:\n  geom: {w_geom}\n  tip : {w_tip}\n"
                f"  先跑训练: python train_pose.py --task both")
    elif args.weights:
        w_geom = w_tip = None
        w_single = Path(args.weights)
    else:
        _t, _g, _s = resolve_default_weights()
        if _t and _g:
            w_tip, w_geom, w_single = _t, _g, None
        elif _s:
            w_geom = w_tip = None
            w_single = _s
        elif _t:
            w_tip, w_geom, w_single = _t, None, None
        elif _g:
            w_tip, w_geom, w_single = None, _g, None
        else:
            raise SystemExit(
                f"找不到任何 pose 权重。期望其中之一:\n"
                f"  双模型: {DEFAULT_WEIGHTS_TIP} + {DEFAULT_WEIGHTS_GEOM}\n"
                f"  单模型: {DEFAULT_WEIGHTS}\n"
                f"  先跑训练: python train_pose.py --task both")

    if dual_wanted:
        w_single = None

    models = {}
    for label, wp in (("geom", w_geom), ("tip", w_tip), ("single", w_single)):
        if wp is None:
            continue
        if not Path(wp).is_file():
            raise SystemExit(f"找不到权重: {wp}")
        print(f"[load] {label:6s} {wp}")
        t0 = time.perf_counter()
        m = _load_model(str(wp))
        info = model_info(m, label)
        print(f"[load] {label:6s} done in {time.perf_counter() - t0:.1f}s   "
              f"arch={info['arch']}  params={info['params_m']}M  "
              f"kpt_shape={info['kpt_shape'] or '?'}")
        models[label] = m

    if "single" in models:
        mode = f"单模型 ({models['single'] and model_info(models['single'])['n_kpt']} 点)"
    else:
        _nt = model_info(models["tip"])["n_kpt"] if "tip" in models else 0
        _ng = model_info(models["geom"])["n_kpt"] if "geom" in models else 0
        mode = f"双模型 (tip {_nt} 点 + geom {_ng} 点 -> 合并 6 点)"
    print(f"[load] 模式: {mode}   imgsz={args.imgsz or '(ultralytics 默认 640)'}")

    if args.debug_dir:
        os.makedirs(args.debug_dir, exist_ok=True)

    # 中文路径
    try:
        sys.path.insert(0, str(ROOT.parent))
        from cv_utils import imread_any
    except Exception:
        imread_any = lambda p: cv2.imread(p)

    files = _gather(args.input)
    if not files:
        raise SystemExit(f"无可用图: {args.input}")

    n_ok = 0
    n_kpt_seen = set()
    n_state = {}
    csv_rows = []
    for p in files:
        bgr = imread_any(p)
        if bgr is None:
            print(f"  [FAIL] {os.path.basename(p)}")
            continue
        overlay = bgr.copy() if args.debug_dir else None
        r = classify(bgr,
                     model=models.get("single"),
                     model_tip=models.get("tip"),
                     model_geom=models.get("geom"),
                     shift=args.shift, conf_thres=args.conf,
                     draw_overlay=overlay, draw_center=args.draw_center,
                     imgsz=(args.imgsz or None), want_state=args.state)
        if r["n_kpt"]:
            n_kpt_seen.add(r["n_kpt"])
        ok = "✓" if r["ok"] else "✗"
        tip = r["tip"] if r["tip"] else "-"
        tc = "-" if r["tip_conf"] is None else f"{r['tip_conf']:.2f}"
        cx = r["cx"] if r["cx"] is not None else "-"
        cy = r["cy"] if r["cy"] is not None else "-"
        a = "-" if r["angle"] is None else f"{r['angle']:6.1f}"
        st_line = ""
        if args.state:
            short, long_ = describe_state(r)
            n_state[short] = n_state.get(short, 0) + 1
            st_line = f"\n        压力状态 = {long_}"
            sp = r.get("state_span") or {}
            csv_rows.append([
                os.path.basename(p), r.get("state") or "未判定",
                sp.get("lo_name", ""), sp.get("lo_deg", ""),
                sp.get("hi_name", ""), sp.get("hi_deg", ""),
                (r.get("state_detail") or {}).get("tip_deg", ""),
                r.get("state_margin_deg", ""),
                round(float(a), 1) if a.strip() != "-" else "",
                r.get("state_reason") or ""])
        print(f"  {ok} {os.path.basename(p):34s} "
              f"tip={str(tip):>16s} c={tc:>5s}  "
              f"kpts={r['n_kpt'] or 0}  center=({cx},{cy})  angle={a}°"
              f"[{r.get('reason') or 'ok'}]{st_line}")
        if r["ok"]:
            n_ok += 1
        if args.debug_dir:
            base = Path(p).stem
            cv2.imwrite(os.path.join(args.debug_dir, base + ".tip.jpg"),
                        overlay)

    kpt_desc = "/".join(str(k) for k in sorted(n_kpt_seen)) or "?"
    print(f"\n  {n_ok}/{len(files)} 张找到针尖   （合并后关键点数: {kpt_desc}）")
    if n_state:
        print("  压力状态分布: "
              + "  ".join(f"{k} {v}" for k, v in sorted(n_state.items())))

    if args.state_csv and csv_rows:
        with open(args.state_csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.writer(fh)
            w.writerow(["文件", "压力状态", "下边界", "下边界角度", "上边界",
                        "上边界角度", "针尖角度", "离最近边界", "圆心针尖夹角",
                        "判不了的原因"])
            w.writerows(csv_rows)
        print(f"  状态明细已写出: {args.state_csv}")


if __name__ == "__main__":
    main()
