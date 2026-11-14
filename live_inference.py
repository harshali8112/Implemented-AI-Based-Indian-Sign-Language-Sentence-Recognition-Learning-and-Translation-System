"""Standalone webcam inference using the same preprocessing as the Flask app."""

from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch

from keypoint_features import SEQUENCE_LENGTH, HolisticFeatureExtractor, process_position_sequence
from train_keypoint_model import KeypointSequenceModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MOTION_THRESHOLD = 0.009
COOLDOWN_SECONDS = 1.5


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    labels = checkpoint.get("labels") or checkpoint.get("id2label")
    if isinstance(labels, dict):
        labels = [str(labels[index]) for index in sorted(labels, key=lambda value: int(value))]
    else:
        labels = [str(label) for label in labels]
    model = KeypointSequenceModel(len(labels)).to(DEVICE)
    state_key = "model_state_dict" if "model_state_dict" in checkpoint else "model_state"
    model.load_state_dict(checkpoint[state_key])
    model.eval()
    return model, labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("isl_keypoint_model.pth"))
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    model, labels = load_checkpoint(args.checkpoint)
    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam")

    positions = deque(maxlen=SEQUENCE_LENGTH)
    probabilities = deque(maxlen=6)
    last_lock = 0.0
    locked = None
    try:
        with HolisticFeatureExtractor() as extractor:
            while True:
                success, frame = camera.read()
                if not success:
                    continue
                positions.append(extractor.frame_to_feature(frame)[:235])
                status = "Idle"
                confidence = 0.0
                if len(positions) == SEQUENCE_LENGTH:
                    position_array = np.asarray(positions, dtype=np.float32)
                    delta = np.diff(position_array, axis=0, prepend=position_array[:1])
                    motion = float(np.mean(np.abs(delta[-1, :126])))
                    if motion <= MOTION_THRESHOLD:
                        status = "Holding stance"
                        probabilities.clear()
                    else:
                        features = process_position_sequence(position_array)
                        with torch.no_grad():
                            current = torch.softmax(model(torch.from_numpy(features).unsqueeze(0).to(DEVICE)), dim=1)[0].cpu().numpy()
                        probabilities.append(current)
                        smoothed = np.mean(np.asarray(probabilities), axis=0)
                        index = int(np.argmax(smoothed))
                        locked = labels[index]
                        confidence = float(smoothed[index])
                        status = "Active prediction"
                        if confidence >= 0.86 and time.monotonic() - last_lock >= COOLDOWN_SECONDS:
                            print(f"Locked: {locked} ({confidence:.2%})")
                            last_lock = time.monotonic()
                            positions.clear()
                            probabilities.clear()
                cv2.putText(frame, f"{status}  {confidence:.1%}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 235, 150), 2, cv2.LINE_AA)
                if locked:
                    cv2.putText(frame, locked[:48], (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.imshow("ISL Keypoint Inference", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
