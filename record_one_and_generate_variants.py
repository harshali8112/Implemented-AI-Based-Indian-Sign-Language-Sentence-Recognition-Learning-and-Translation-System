"""Record one source clip then generate 19 realistic dataset variations for the same sentence.

Ensures exact consistency with the existing custom_dataset clips:
- Resolution: 640x480
- Frame Rate: 20.0 FPS
- Duration: 7.0 seconds (~140 frames)
- Codec: FMP4 / XVID in .avi format
- Total clips per class: 20 (1 base recording + 19 augmented variants)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record a single base clip and create 19 dataset variants for the same sentence."
    )
    parser.add_argument(
        "--sentence",
        type=str,
        default="",
        help="Sentence label directory, e.g. 'thank you'. Prompts if omitted.",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=7.0,
        help="Seconds for the base video clip (default: 7.0s to match existing dataset).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Frames per second for recorded and generated clips (default: 20.0).",
    )
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index to use for the base clip.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Frame width (default: 640).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Frame height (default: 480).",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=1.0,
        help="Seconds to pause before generating variants.",
    )
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


def save_video(
    video_path: Path,
    frames: list[np.ndarray],
    fps: float = 20.0,
    width: int = 640,
    height: int = 480,
) -> None:
    if not frames:
        raise RuntimeError("No frames to save.")

    # Try FMP4 first (matches existing dataset), fallback to XVID
    fourcc = cv2.VideoWriter_fourcc(*"FMP4")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {video_path}")

    try:
        for frame in frames:
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            writer.write(frame)
    finally:
        writer.release()


def record_single_clip(
    video_path: Path,
    camera_index: int,
    duration: float,
    fps: float,
    width: int = 640,
    height: int = 480,
) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam index {camera_index} is not available.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    fourcc = cv2.VideoWriter_fourcc(*"FMP4")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not initialize video writer for {video_path}.")

    print(f"\n=======================================================")
    print(f"[*] Preparing to record base clip: {video_path}")
    print(f"Target format: {width}x{height} @ {fps} FPS, {duration:.1f} seconds")
    print(f"[*] Live camera preview is active. Position yourself, then press SPACEBAR to begin!")
    print(f"=======================================================")

    try:
        # Step 1: Preview mode - wait for user to be ready
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Sign: {video_path.parent.name}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                "Press SPACEBAR when ready",
                (int(width * 0.15), int(height * 0.5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                preview,
                "(ESC to cancel)",
                (int(width * 0.38), int(height * 0.6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                1,
            )
            cv2.imshow("Webcam Recording", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (32, 13):  # SPACEBAR or ENTER
                break
            if key == 27:  # ESC
                raise KeyboardInterrupt

        # Step 2: 3-Second countdown so the user can raise hands properly
        for countdown in range(3, 0, -1):
            countdown_start = time.time()
            while time.time() - countdown_start < 1.0:
                ret, frame = cap.read()
                if not ret:
                    break
                display = frame.copy()
                cv2.putText(
                    display,
                    f"Get Ready: {countdown}",
                    (int(width * 0.28), int(height * 0.5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.4,
                    (0, 255, 255),
                    3,
                )
                cv2.imshow("Webcam Recording", display)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    raise KeyboardInterrupt

        print(f"\n[REC] RECORDING NOW! Perform the sign for {duration:.1f}s...")
        start_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frame from webcam.")

            elapsed = time.time() - start_time
            remaining = max(0.0, duration - elapsed)

            display = frame.copy()
            cv2.putText(
                display,
                f"Recording... {remaining:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Webcam Recording", display)

            # Save clean frame to match dataset
            out.write(frame)

            if elapsed >= duration:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                raise KeyboardInterrupt
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    print(f"[OK] Successfully saved base clip: {video_path}")


# -------------------- Real-World Video Augmentation Functions --------------------


def apply_speed_change(frames: list[np.ndarray], speed_factor: float) -> list[np.ndarray]:
    """Change motion speed:
    speed_factor > 1.0: Faster signing (fewer frames at same FPS)
    speed_factor < 1.0: Slower signing (more frames at same FPS)
    """
    if len(frames) <= 2:
        return frames[:]
    desired_length = max(2, int(round(len(frames) / speed_factor)))
    indices = np.linspace(0, len(frames) - 1, desired_length).astype(int)
    return [frames[i] for i in indices]


def apply_horizontal_shift(frame: np.ndarray, dx: int) -> np.ndarray:
    """Shift horizontally with natural edge replication (avoids harsh black borders)."""
    h, w = frame.shape[:2]
    shifted = np.empty_like(frame)
    if dx > 0:
        shifted[:, dx:] = frame[:, : w - dx]
        shifted[:, :dx] = frame[:, :1]
    elif dx < 0:
        shifted[:, : w + dx] = frame[:, -dx:]
        shifted[:, w + dx :] = frame[:, -1:]
    else:
        return frame.copy()
    return shifted


def apply_zoom(frame: np.ndarray, scale: float) -> np.ndarray:
    """Zoom in/out with clean border handling."""
    h, w = frame.shape[:2]
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if scale > 1.0:
        # Zoom in: crop center
        x1 = (new_w - w) // 2
        y1 = (new_h - h) // 2
        return resized[y1 : y1 + h, x1 : x1 + w]
    elif scale < 1.0:
        # Zoom out: pad borders
        padded = np.empty_like(frame)
        x1 = (w - new_w) // 2
        y1 = (h - new_h) // 2
        padded[y1 : y1 + new_h, x1 : x1 + new_w] = resized
        if y1 > 0:
            padded[:y1, :] = padded[y1 : y1 + 1, :]
            padded[y1 + new_h :, :] = padded[y1 + new_h - 1 : y1 + new_h, :]
        if x1 > 0:
            padded[:, :x1] = padded[:, x1 : x1 + 1]
            padded[:, x1 + new_w :] = padded[:, x1 + new_w - 1 : x1 + new_w]
        return padded
    return frame.copy()


def apply_lighting(frame: np.ndarray, alpha: float, beta: int) -> np.ndarray:
    """Simulate variations in ambient room lighting / contrast."""
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)


def generate_variants(source_path: Path, sentence_dir: Path, target_total: int = 20) -> None:
    """Generates remaining variants so sentence_dir contains exactly target_total (20) clips."""
    base_frames = read_frames(source_path)
    sentence_dir.mkdir(parents=True, exist_ok=True)

    # 19 tailored variants that naturally relate to previous recordings:
    # 4 slow, 4 fast, 3 shifted left, 3 shifted right, 2 zoom/distance, 3 lighting
    specs: list[tuple[str, dict]] = [
        # Slow signing (speed < 1.0)
        ("Slow (0.80x speed)", {"type": "speed", "factor": 0.80}),
        ("Slow (0.85x speed)", {"type": "speed", "factor": 0.85}),
        ("Slow (0.90x speed)", {"type": "speed", "factor": 0.90}),
        ("Slow (0.95x speed)", {"type": "speed", "factor": 0.95}),
        # Fast signing (speed > 1.0)
        ("Fast (1.10x speed)", {"type": "speed", "factor": 1.10}),
        ("Fast (1.20x speed)", {"type": "speed", "factor": 1.20}),
        ("Fast (1.30x speed)", {"type": "speed", "factor": 1.30}),
        ("Fast (1.40x speed)", {"type": "speed", "factor": 1.40}),
        # Shifted Left (user seated slightly to the right of frame)
        ("Shift Left (-20px)", {"type": "shift", "dx": -20}),
        ("Shift Left (-35px)", {"type": "shift", "dx": -35}),
        ("Shift Left (-50px)", {"type": "shift", "dx": -50}),
        # Shifted Right (user seated slightly to the left of frame)
        ("Shift Right (+20px)", {"type": "shift", "dx": 20}),
        ("Shift Right (+35px)", {"type": "shift", "dx": 35}),
        ("Shift Right (+50px)", {"type": "shift", "dx": 50}),
        # Distance / Zoom variation (sitting closer or farther)
        ("Zoom In (1.06x scale)", {"type": "zoom", "scale": 1.06}),
        ("Zoom Out (0.94x scale)", {"type": "zoom", "scale": 0.94}),
        # Lighting / Camera exposure variations
        ("Bright Lighting (+15% contrast)", {"type": "lighting", "alpha": 1.15, "beta": 15}),
        ("Dim Lighting (-12% brightness)", {"type": "lighting", "alpha": 0.88, "beta": -15}),
        ("High Contrast", {"type": "lighting", "alpha": 1.25, "beta": -10}),
    ]

    print(f"\n[+] Generating {len(specs)} augmented clips from {source_path.name}...")

    # We want files to be 2.avi, 3.avi, ..., 20.avi
    for i, (description, config) in enumerate(specs, start=2):
        out_path = sentence_dir / f"{i}.avi"
        op_type = config["type"]

        if op_type == "speed":
            variant_frames = apply_speed_change(base_frames, config["factor"])
        elif op_type == "shift":
            dx = config["dx"]
            variant_frames = [apply_horizontal_shift(f, dx) for f in base_frames]
        elif op_type == "zoom":
            scale = config["scale"]
            variant_frames = [apply_zoom(f, scale) for f in base_frames]
        elif op_type == "lighting":
            alpha = config["alpha"]
            beta = config["beta"]
            variant_frames = [apply_lighting(f, alpha, beta) for f in base_frames]
        else:
            variant_frames = base_frames[:]

        save_video(out_path, variant_frames, fps=20.0, width=640, height=480)
        print(f"  [{i:02d}/{target_total}] Saved {out_path.name:<8} -> {description}")

    total_files = len(list(sentence_dir.glob("*.avi")))
    print(f"\n[OK] Completed! Folder '{sentence_dir}' now contains {total_files} clips (1.avi to {total_files}.avi).")


def main() -> None:
    args = parse_args()
    sentence = (args.sentence or input("Enter sentence / sign name: ")).strip()
    if not sentence:
        raise ValueError("Sentence name cannot be empty.")

    sentence_dir = Path("custom_dataset") / sentence
    sentence_dir.mkdir(parents=True, exist_ok=True)

    base_path = sentence_dir / "1.avi"
    if not base_path.exists() or base_path.stat().st_size == 0:
        if base_path.exists():
            base_path.unlink()
        print(f"No valid base clip found for '{sentence}'. Initiating webcam recording...")
        try:
            record_single_clip(
                base_path,
                args.camera_index,
                args.duration,
                args.fps,
                args.width,
                args.height,
            )
        except KeyboardInterrupt:
            print("\n[!] Recording cancelled by user.")
            if base_path.exists() and base_path.stat().st_size == 0:
                base_path.unlink()
            return
    else:
        print(f"Found existing base clip: {base_path}")

    if args.pause > 0:
        time.sleep(args.pause)

    generate_variants(base_path, sentence_dir, target_total=20)


if __name__ == "__main__":
    main()
