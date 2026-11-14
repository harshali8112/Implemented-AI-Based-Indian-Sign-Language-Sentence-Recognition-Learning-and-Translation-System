"""Extract fixed-length MediaPipe Holistic sequences from recorded videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from keypoint_features import HolisticFeatureExtractor, process_position_sequence


def extract_video(video_path: Path, extractor: HolisticFeatureExtractor) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frames = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(extractor.frame_to_feature(frame)[:235])
    finally:
        capture.release()

    if not frames:
        raise RuntimeError(f"No frames found in {video_path}")
    return process_position_sequence(np.asarray(frames, dtype=np.float32))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--videos", type=Path, default=Path("custom_dataset"))
    parser.add_argument("--output", type=Path, default=Path("keypoint_sequences"))
    parser.add_argument("--overwrite", action="store_true", help="Force re-extraction of all videos even if .npy exists.")
    args = parser.parse_args()

    video_paths = sorted(path for path in args.videos.rglob("*.avi") if path.is_file())
    if not video_paths:
        raise FileNotFoundError(f"No .avi videos found below {args.videos}")
    args.output.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    with HolisticFeatureExtractor() as extractor:
        for video_path in video_paths:
            label = video_path.parent.name
            destination = args.output / label / f"{video_path.stem}.npy"
            if destination.exists() and not args.overwrite:
                skipped += 1
                continue
            try:
                sequence = extract_video(video_path, extractor)
                destination.parent.mkdir(parents=True, exist_ok=True)
                np.save(destination, sequence)
                processed += 1
                print(f"[{processed}/{len(video_paths) - skipped}] {label}/{video_path.name} -> {sequence.shape}")
            except Exception as error:
                failed += 1
                print(f"Skipped {video_path}: {error}")

    print(f"Processed {processed} videos; skipped {skipped} already existing; failed {failed}; output: {args.output}")


if __name__ == "__main__":
    main()
