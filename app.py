# -*- coding: utf-8 -*-
"""
Веб-бот: загрузите фото на странице — получите выровненное и улучшенное.

Локальный запуск:  python app.py   ->  http://127.0.0.1:5000
На Render.com:     gunicorn app:app --bind 0.0.0.0:$PORT
"""
import base64
import os
import time

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import bot_processor

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 МБ

MAX_SIDE = 3000  # большие изображения уменьшаются до этой стороны


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"ok": True})


@app.route("/api/process", methods=["POST"])
def api_process():
    if "image" not in request.files or request.files["image"].filename == "":
        return jsonify({"ok": False, "error": "Файл не выбран"}), 400

    raw = request.files["image"].read()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"ok": False,
                        "error": "Формат не поддерживается (нужен JPG или PNG)"}), 400

    h, w = img.shape[:2]
    if max(h, w) > MAX_SIDE:
        k = MAX_SIDE / float(max(h, w))
        img = cv2.resize(img, (int(round(w * k)), int(round(h * k))),
                         interpolation=cv2.INTER_AREA)

    opts = {
        "do_perspective": request.form.get("perspective", "1") == "1",
        "do_skew": request.form.get("skew", "1") == "1",
        "do_enhance": request.form.get("enhance", "1") == "1",
    }

    t0 = time.time()
    try:
        result, info = bot_processor.process_image(img, **opts)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": "Ошибка обработки: %s" % exc}), 500

    ok, buf = cv2.imencode(".jpg", result, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return jsonify({"ok": False, "error": "Ошибка кодирования результата"}), 500

    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return jsonify({
        "ok": True,
        "image": "data:image/jpeg;base64," + b64,
        "info": info,
        "seconds": round(time.time() - t0, 1),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("Бот запущен: http://127.0.0.1:%d" % port)
    app.run(host="0.0.0.0", port=port)
