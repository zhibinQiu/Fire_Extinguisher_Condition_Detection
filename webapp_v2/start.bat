@echo off
REM ============================================================
REM  Gauge OCR - Web UI one-click launcher (Windows)
REM  Pipeline: YOLO detect -> box crop -> ellipse-to-circle rectify
REM            -> YOLO-pose 6 keypoints
REM            (kpt0 center / kpt1-4 zone boundaries / kpt5 needle tip)
REM  Opens http://127.0.0.1:8770/ in the default browser.
REM  Close this window (or press Ctrl+C) to stop the server.
REM
REM  NOTE: step-2 pose needs ultralytics (+torch), so this launcher
REM        prefers the conda env "wanhua" over a bare "python" on PATH.
REM  2026-09-17: the UI shows the 6 keypoints AND judges the pressure
REM              state (under/normal/over) from them. See server.py header.
REM  2026-09-17 later: the crop is rectified to a circle BEFORE pose
REM              (trans/dial_ellipse_to_circle.py) -- a side-view dial is
REM              an ellipse, and the center->tip angle is anisotropically
REM              compressed on it, so the angle-based state rule was
REM              comparing wrong angles. Falls back to the raw crop when
REM              rectification fails. Disable with --no-rectify.
REM  2026-09-17 later: tip model swapped to yolov8n-pose
REM              (classifier/yolo-v8n-pose-best.pt, 3.08M, kpt_shape [1,3]).
REM              Geometry model is still yolov8m-pose (classifier/best_geom.pt,
REM              26.4M, [5,3]). Arch/params are read from the ckpt, so the UI
REM              reports whatever is on disk; override with --weights-tip.
REM ============================================================
setlocal
set "DIR=%~dp0"
set "PORT=8770"
set "PY="

if exist "E:\Anaconda\envs\wanhua\python.exe" set "PY=E:\Anaconda\envs\wanhua\python.exe"

if not defined PY (
    for %%I in (python.exe) do set "PY=%%~$PATH:I"
)
if not defined PY (
    echo [ERROR] python not found.
    echo         Install Python, or fix the conda path in this file.
    pause
    exit /b 1
)

echo [1/3] Python : %PY%
echo [2/3] Deps   : checking ultralytics / torch / cv2 ...
"%PY%" -c "import torch,cv2,ultralytics;print('       ok  torch',torch.__version__,' ultralytics',ultralytics.__version__)"
if errorlevel 1 (
    echo        [ERROR] ultralytics / torch / cv2 missing in this interpreter.
    echo        Run:  "%PY%" -m pip install -r "%~dp0requirements.txt"
    pause
    exit /b 1
)
echo [3/3] Server : %DIR%server.py   port %PORT%
echo.
"%PY%" "%DIR%server.py" --port %PORT% --open
echo.
echo Server stopped.
pause
