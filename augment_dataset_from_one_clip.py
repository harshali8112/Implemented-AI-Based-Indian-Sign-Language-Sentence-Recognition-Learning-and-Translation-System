"""Generate extra sign-language clips from a single reference video."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create multiple dataset variants from one good reference video."
    )
    parser.add_argument("--source", type=Path, required=True, help="One source .avi clip to use as the template.")
    parser.add_argument("--count", type=int, default=10, help="How many extra clips to generate.")
    parser.add_argument("--sentence", type=str, default="", help="Optional sentence folder name. If omitted, the source parent directory is used.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Destination folder. Defaults to custom_dataset/<sentence>.")
    parser.add_argument("--fps", type=float, default=20.0, help="Target FPS for generated clips.")
    parser.add_argument("--width", type=int, default=640, help="Output width.")
    parser.add_argument("--height", type=int, default=480, help="Output height.")
    return parser.parse_args()


def read_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames found in {video_path}")
    return frames


def apply_brightness_and_contrast(frame: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def apply_small_shift(frame: np.ndarray, dx: int, dy: int) -> np.ndarray:
    height, width = frame.shape[:2]
    shifted = np.zeros_like(frame)
    x_start = max(0, dx)
    y_start = max(0, dy)
    src_x = max(0, -dx)
    src_y = max(0, -dy)
    src_w = width - abs(dx)
    src_h = height - abs(dy)
    if src_w <= 0 or src_h <= 0:
        return frame
    shifted[y_start:y_start + src_h, x_start:x_start + src_w] = frame[src_y:src_y + src_h, src_x:src_x + src_w]
    return shifted


def apply_zoom(frame: np.ndarray, scale: float) -> np.ndarray:
    height, width = frame.shape[:2]
    new_w = max(1, int(round(width * scale)))
    new_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, (new_w - width) // 2)
    y1 = max(0, (new_h - height) // 2)
    x2 = x1 + width
    y2 = y1 + height
    return resized[y1:y2, x1:x2]


def apply_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 1:
        return frame
    return cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)


def augment_frame(frame: np.ndarray, seed: int) -> np.ndarray:
    rng = random.Random(seed)

    alpha = rng.uniform(0.85, 1.2)
    beta = rng.randint(-30, 30)
    frame = apply_brightness_and_contrast(frame, alpha, beta)

    dx = rng.randint(-18, 18)
    dy = rng.randint(-18, 18)
    frame = apply_small_shift(frame, dx, dy)

    scale = rng.uniform(0.95, 1.08)
    frame = apply_zoom(frame, scale)

    if rng.random() < 0.7:
        kernel_size = rng.choice([3, 5, 7])
        frame = apply_blur(frame, kernel_size)

    return frame


def find_next_index(folder: Path) -> int:
    existing = sorted(folder.glob("*.avi"))
    return len(existing) + 1


def save_video(video_path: Path, frames: list[np.ndarray], fps: float, width: int, height: int) -> None:
    if not frames:
        raise RuntimeError("No frames to save.")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {video_path}")

    try:
        for frame in frames:
            resized = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(resized)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source video not found: {source}")

    sentence = args.sentence.strip() if args.sentence.strip() else source.parent.name
    destination_root = args.output_dir.resolve() if args.output_dir else Path("custom_dataset") / sentence
    destination_root.mkdir(parents=True, exist_ok=True)

    frames = read_frames(source)
    if args.count <= 0:
        raise ValueError("--count must be greater than 0.")

    start_index = find_next_index(destination_root)
    print(f"Generating {args.count} augmented clips for sentence '{sentence}' in {destination_root}")
    print(f"Source: {source}")

    for i in range(args.count):
        variant_frames = [augment_frame(frame.copy(), (start_index + i) * 97 + j) for j, frame in enumerate(frames)]
        target_path = destination_root / f"{start_index + i}.avi"
        save_video(target_path, variant_frames, args.fps, args.width, args.height)
        print(f"[{i + 1}/{args.count}] Saved {target_path}")

    print(f"\n✅ Created {args.count} new clips from {source}.")


if __name__ == "__main__":
    main()
