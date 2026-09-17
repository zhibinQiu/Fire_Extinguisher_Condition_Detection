# test-device-ocr

灭火器压力表识别

## 跑起来

Windows：双击 `webapp_v2/start.bat`（自动找 `E:\Anaconda\envs\wanhua\python.exe`，
找不到就退回 PATH 上的 python），浏览器打开 <http://127.0.0.1:8770/>。

命令行：

```bat
E:\Anaconda\envs\wanhua\python.exe webapp_v2\server.py --port 8770 --open
```

换台机器先装依赖：

```bat
python -m pip install -r webapp_v2\requirements.txt
```

## 目录结构

```
test-device-ocr/
├── cv_utils.py                       中文路径 imread/imwrite（被下面多个模块 import）
├── classifier/
│   ├── classify_image_yolo.py        第一步：YOLO 表盘检测封装
│   ├── classify_pose.py              第二步：YOLO-pose 关键点 + 压力状态判据
│   ├── yolo-v8n-pose-best.pt         针尖权重  yolov8n-pose  3.08M  kpt_shape [1,3]
│   └── best_geom.pt                  几何权重  yolov8m-pose 26.42M  kpt_shape [5,3]
├── trans/
│   └── dial_ellipse_to_circle.py     椭圆 -> 正圆校正（裁切与 pose 之间）
├── tools/
│   ├── gauge_box_crop.py             按检测框直裁（clamp_box / crop_box）
│   └── gauge_circle_crop.py          割圆相关工具 + 插值/锐化选项
├── yolo_pipeline/output/
│   └── best.pt                       第一步检测权重  yolov8 detect  3.02M  imgsz=1216
└── webapp_v2/
    ├── server.py                     服务端（只用标准库 http.server）
    ├── index.html                    单页前端（无外部 CDN，全部内联）
    ├── start.bat / start.sh          一键启动
    └── requirements.txt              运行时依赖
```

⚠ 目录层级不能改：所有脚本都用 `BASE = dirname(dirname(__file__))` 定位包根，
再把 `classifier/ yolo_pipeline/ tools/ trans/` 塞进 `sys.path`，默认权重也写成
`{BASE}/classifier/xxx.pt`。层级一动，路径解析就全错。

## 流水线

1. **YOLO 检出表盘** `--det_imgsz 1216`（= 检测器训练分辨率）
2. **按检测框直裁** 960×960
3. **椭圆→正圆校正**（侧视/斜拍时表盘是椭圆，角度被非等比压缩；不校正则三态判据
   比的是错的角度）。校正失败或判定不可信时**自动降级**用直裁图，不丢结果。
4. **YOLO-pose 出 6 个关键点**（双模型合并）
5. **判压力状态** —— 看「圆心→针尖」射线被夹在哪两条相邻边界射线之间

### 6 个关键点的口径（硬约定）

| 槽位 | 含义 | 来源 |
|---|---|---|
| kpt0 | 表盘圆心 | geom |
| kpt1 | 欠压边界（红区起点） | geom |
| kpt2 | 欠压 / 正常 分界 | geom |
| kpt3 | 正常 / 超压 分界 | geom |
| kpt4 | 超压边界（黄区终点） | geom |
| kpt5 | 指针尖端 | tip |

**kpt0 恒为圆心、最后一个恒为针尖**，下游都按这个写的，换点位方案时不要动头尾。


## 接口

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/` | 页面 |
| GET | `/api/health` | 健康检查 + 当前权重/参数（含两个模型各自的 arch/kpt_shape） |
| GET | `/api/config` | 页面初始化默认参数 |
| POST | `/api/predict` | 图片字节流 → JSON |
