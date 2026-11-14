"""Flask web server for real-time ISL keypoint inference."""

from __future__ import annotations

import os
import threading
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, Response, jsonify, render_template, request

from keypoint_features import HolisticFeatureExtractor, SEQUENCE_LENGTH, process_position_sequence
from train_keypoint_model import KeypointSequenceModel

BASE_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = BASE_DIR / "isl_keypoint_model.pth"
MOTION_THRESHOLD = 0.009
DEFAULT_CONFIDENCE_THRESHOLD = 0.86
SPECIAL_THRESHOLDS = {"are you hiding something": 0.91}
COOLDOWN_SECONDS = 1.5
DISPLAY_HOLD_SECONDS = 3.0
CAMERA_INDEX = 0

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_checkpoint():
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    raw_labels = checkpoint.get("labels")
    if raw_labels is None:
        raw_labels = checkpoint.get("id2label")
    if isinstance(raw_labels, dict):
        labels = [str(raw_labels[index]) for index in sorted(raw_labels, key=lambda value: int(value))]
    else:
        labels = [str(label) for label in raw_labels]
    model = KeypointSequenceModel(len(labels)).to(DEVICE)
    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model, labels


class InferenceWorker:
    def __init__(self) -> None:
        self.model, self.labels = load_model_checkpoint()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.stop_event = threading.Event()
        self.frame_buffer = deque(maxlen=SEQUENCE_LENGTH)
        self.probability_buffer = deque(maxlen=6)
        self.locked_tokens: list[dict] = []
        self.latest_jpeg: bytes | None = None
        self.status = "Idle"
        self.prediction = ""
        self.confidence = 0.0
        self.frame_count = 0
        self.last_lock_time = 0.0
        self.display_until = 0.0
        self.display_prediction = ""
        self.display_confidence = 0.0
        self.camera = None
        self.thread = threading.Thread(target=self._run, name="isl-camera-worker", daemon=True)

    def start(self) -> None:
        if not self.thread.is_alive():
            self.thread.start()

    def reset_sentence(self) -> None:
        with self.condition:
            self.locked_tokens.clear()
            self.frame_buffer.clear()
            self.probability_buffer.clear()
            self.prediction = ""
            self.confidence = 0.0
            self.status = "Idle"
            self.last_lock_time = 0.0
            self.display_until = 0.0
            self.display_prediction = ""
            self.display_confidence = 0.0
            self.condition.notify_all()

    def snapshot(self, cursor: int = 0) -> dict:
        with self.lock:
            display_active = time.monotonic() < self.display_until
            display_status = "Sentence locked" if display_active else self.status
            display_prediction = self.display_prediction if display_active else self.prediction
            display_confidence = self.display_confidence if display_active else self.confidence
            return {
                "status": display_status,
                "prediction": display_prediction,
                "confidence": display_confidence,
                "frame_count": self.frame_count,
                "locked_tokens": self.locked_tokens[cursor:],
                "token_count": len(self.locked_tokens),
                "motion_threshold": MOTION_THRESHOLD,
            }

    def mjpeg_frames(self):
        self.start()
        while not self.stop_event.is_set():
            with self.condition:
                self.condition.wait_for(lambda: self.latest_jpeg is not None or self.stop_event.is_set(), timeout=1.0)
                jpeg = self.latest_jpeg
            if jpeg is None:
                continue
            yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"

    def _run(self) -> None:
        try:
            self.camera = cv2.VideoCapture(CAMERA_INDEX)
            if not self.camera.isOpened():
                with self.lock:
                    self.status = "Camera unavailable"
                return
            with HolisticFeatureExtractor() as extractor:
                while not self.stop_event.is_set():
                    success, frame = self.camera.read()
                    if not success:
                        time.sleep(0.03)
                        continue
                    position = extractor.frame_to_feature(frame)[:235]
                    self._update_prediction(position)
                    self._publish_frame(frame)
        except Exception as error:
            app.logger.exception("Inference worker stopped")
            with self.condition:
                self.status = f"Error: {error}"
                self.condition.notify_all()
        finally:
            if self.camera is not None:
                self.camera.release()

    def _update_prediction(self, position: np.ndarray) -> None:
        with self.condition:
            self.frame_buffer.append(position)
            self.frame_count = len(self.frame_buffer)
            if len(self.frame_buffer) < SEQUENCE_LENGTH:
                if time.monotonic() >= self.display_until:
                    self.status = "Idle"
                    self.prediction = ""
                    self.confidence = 0.0
                self.condition.notify_all()
                return

            positions = np.asarray(self.frame_buffer, dtype=np.float32)
            delta = np.diff(positions, axis=0, prepend=positions[:1])
            motion_energy = float(np.mean(np.abs(delta[-1, :126])))
            if motion_energy <= MOTION_THRESHOLD:
                if time.monotonic() >= self.display_until:
                    self.status = "Holding stance"
                    self.prediction = ""
                    self.confidence = 0.0
                self.probability_buffer.clear()
                self.condition.notify_all()
                return

            features = process_position_sequence(positions)
            inputs = torch.from_numpy(features).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                probabilities = torch.softmax(self.model(inputs), dim=1)[0].cpu().numpy()
            self.probability_buffer.append(probabilities)
            smoothed = np.mean(np.asarray(self.probability_buffer), axis=0)
            class_index = int(np.argmax(smoothed))
            self.prediction = self.labels[class_index]
            self.confidence = float(smoothed[class_index])
            self.status = "Active prediction"
            threshold = SPECIAL_THRESHOLDS.get(self.prediction, DEFAULT_CONFIDENCE_THRESHOLD)
            now = time.monotonic()
            if self.confidence >= threshold and now - self.last_lock_time >= COOLDOWN_SECONDS:
                token = {"text": self.prediction, "confidence": self.confidence, "timestamp": time.time()}
                self.locked_tokens.append(token)
                self.last_lock_time = now
                self.display_prediction = self.prediction
                self.display_confidence = self.confidence
                self.display_until = now + DISPLAY_HOLD_SECONDS
                self.frame_buffer.clear()
                self.probability_buffer.clear()
                self.status = "Sentence locked"
            self.condition.notify_all()

    def _publish_frame(self, frame: np.ndarray) -> None:
        with self.lock:
            status = self.status
            prediction = self.prediction
            confidence = self.confidence
        color = (80, 235, 150) if status == "Sentence locked" else (170, 220, 255)
        cv2.rectangle(frame, (12, 12), (470, 76), (12, 18, 28), -1)
        cv2.putText(frame, f"{status}  {confidence * 100:.1f}%", (25, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2, cv2.LINE_AA)
        if prediction:
            cv2.putText(frame, prediction[:42], (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (245, 245, 245), 1, cv2.LINE_AA)
        encoded, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if encoded:
            with self.condition:
                self.latest_jpeg = buffer.tobytes()
                self.condition.notify_all()


worker = None
worker_error = None
try:
    worker = InferenceWorker()
except Exception as error:
    worker_error = str(error)


@app.route("/")
def index():
    return render_template("home.html")


@app.route("/translator")
def translator():
    return render_template("translator.html")


@app.route("/feedback", methods=["GET", "POST"])
def feedback():
    return render_template("feedback.html", submitted=request.method == "POST")


@app.route("/video_feed")
def video_feed():
    if worker is None:
        return jsonify({"success": False, "message": worker_error}), 503
    return Response(worker.mjpeg_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/get_prediction")
def get_prediction():
    if worker is None:
        return jsonify({"success": False, "message": worker_error}), 503
    try:
        cursor = max(0, int(request.args.get("cursor", 0)))
    except ValueError:
        cursor = 0
    return jsonify({"success": True, **worker.snapshot(cursor)})


@app.route("/reset_sentence", methods=["POST"])
def reset_sentence():
    if worker is None:
        return jsonify({"success": False, "message": worker_error}), 503
    worker.reset_sentence()
    return jsonify({"success": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False, threaded=True)
