import os
import sys
import time
import cv2
import numpy as np
from flask import Flask, render_template, request, Response, redirect, url_for

# Fix OpenCV/ffmpeg threading issues
cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)

APP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(APP_DIR, "src")
sys.path.append(SRC_DIR)

from VideoSkeleton import VideoSkeleton
from VideoReader import VideoReader
from Skeleton import Skeleton
from GenVanillaNN import GenVanillaNN
from GenGAN import GenGAN

app = Flask(__name__)

STATE = {
    "running": False,
    "model": None,
    "target_vs": None,
    "source_reader": None,
    "ske": None,
    "stride": 5,
    "gen_type": 4,
    "source_path": None,

    # IMPORTANT: target fixe = taichi1
    "target_path": "data/taichi1.mp4",
}

ALLOWED_EXT = {".mp4", ".avi", ".mov", ".mkv"}


def list_videos_in_data():
    data_dir = os.path.join(APP_DIR, "data")
    vids = []
    if os.path.isdir(data_dir):
        for fn in os.listdir(data_dir):
            ext = os.path.splitext(fn.lower())[1]
            if ext in ALLOWED_EXT:
                vids.append(fn)
    return sorted(vids)


def build_generator(gen_type: int, target_video_abs: str):
    """
    gen_type:
      2 -> VanillaNN (26D vector -> image)        optSkeOrImage=1
      3 -> VanillaNN (stickman image -> image)   optSkeOrImage=2
      4 -> GAN
    """
    target_vs = VideoSkeleton(target_video_abs)

    if gen_type == 2:
        gen = GenVanillaNN(target_vs, loadFromFile=True, optSkeOrImage=1)
    elif gen_type == 3:
        gen = GenVanillaNN(target_vs, loadFromFile=True, optSkeOrImage=2)
    elif gen_type == 4:
        gen = GenGAN(target_vs, loadFromFile=True)
    else:
        raise ValueError("gen_type must be 2, 3 or 4")

    return gen, target_vs


def ensure_uint8_bgr(img, fallback_shape=(256, 256)):
    if img is None:
        return np.zeros((fallback_shape[0], fallback_shape[1], 3), dtype=np.uint8)

    if img.dtype != np.uint8:
        x = img
        if x.max() <= 1.5:
            x = x * 255.0
        img = np.clip(x, 0, 255).astype(np.uint8)

    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


def make_triptych(src_bgr, ske_obj, gen_out_bgr, H=256, W_panel=256):
    src_bgr = cv2.resize(src_bgr, (W_panel, H))
    gen_out_bgr = cv2.resize(gen_out_bgr, (W_panel, H))

    ske_img = np.zeros((H, W_panel, 3), dtype=np.uint8)
    ske_obj.draw(ske_img)

    frame = np.hstack([src_bgr, ske_img, gen_out_bgr])

    cv2.putText(frame, "SOURCE", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "SQUELETTE", (W_panel + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(frame, "GENERATION", (2*W_panel + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)

    return frame


def gen_frames():
    H, W_panel = 256, 256
    err = np.zeros((H, W_panel, 3), dtype=np.uint8)
    err[:] = (0, 0, 255)

    fps_smooth = 0.0
    last_t = time.time()

    while True:
        if not STATE["running"]:
            stopped = np.zeros((H, W_panel * 3, 3), dtype=np.uint8)
            cv2.putText(stopped, "STOPPED", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
            ok, buf = cv2.imencode(".jpg", stopped)
            if not ok:
                break
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.2)
            continue

        reader = STATE["source_reader"]
        if reader is None:
            time.sleep(0.1)
            continue

        # stride: saute des frames pour être plus rapide
        frame = None
        for _ in range(max(1, STATE["stride"])):
            frame = reader.readFrame()
            if frame is None:
                break

        if frame is None:
            STATE["running"] = False
            continue

        isSke, src_crop, STATE["ske"] = STATE["target_vs"].cropAndSke(frame, STATE["ske"])

        if not isSke:
            src_crop = ensure_uint8_bgr(src_crop, (H, W_panel))
            trip = np.hstack([cv2.resize(src_crop, (W_panel, H)), err.copy(), err.copy()])
        else:
            src_crop = ensure_uint8_bgr(src_crop, (H, W_panel))
            out = STATE["model"].generate(STATE["ske"])
            out_bgr = ensure_uint8_bgr(out, (H, W_panel))
            trip = make_triptych(src_crop, STATE["ske"], out_bgr, H=H, W_panel=W_panel)

        # FPS
        t = time.time()
        dt = t - last_t
        last_t = t
        if dt > 0:
            fps_smooth = 0.9 * fps_smooth + 0.1 * (1.0 / dt)
        cv2.putText(trip, f"FPS: {fps_smooth:.0f}", (10, H - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)

        ok, buf = cv2.imencode(".jpg", trip)
        if not ok:
            continue

        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


@app.route("/", methods=["GET"])
def index():
    videos = list_videos_in_data()
    return render_template("index.html", videos=videos, target_fixed=STATE["target_path"])


@app.route("/start", methods=["POST"])
def start():
    gen_type = int(request.form.get("gen_type", "4"))
    stride = int(request.form.get("stride", "5"))
    source_video = request.form.get("source_video", "taichi2.mp4")

    # target FIXE (on ne le modifie plus)
    target_video_abs = os.path.join(APP_DIR, STATE["target_path"])

    source_path = os.path.join(APP_DIR, "data", source_video)
    if not os.path.isfile(source_path):
        return f"Source video not found: {source_path}", 400
    if not os.path.isfile(target_video_abs):
        return f"Target video not found: {target_video_abs}", 400

    # stop ancien run si existant
    STATE["running"] = False
    time.sleep(0.2)

    model, target_vs = build_generator(gen_type, target_video_abs)

    STATE["gen_type"] = gen_type
    STATE["stride"] = max(1, stride)
    STATE["source_path"] = source_path
    STATE["model"] = model
    STATE["target_vs"] = target_vs
    STATE["source_reader"] = VideoReader(source_path)
    STATE["ske"] = Skeleton()
    STATE["running"] = True

    return redirect(url_for("viewer"))


@app.route("/viewer")
def viewer():
    return render_template("viewer.html")


@app.route("/stop")
def stop():
    STATE["running"] = False
    return redirect(url_for("index"))


@app.route("/stream")
def stream():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    # Important pour éviter le crash ffmpeg async_lock
    app.run(host="127.0.0.1", port=5000, debug=True, threaded=False)
