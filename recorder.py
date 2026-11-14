import argparse
import time
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record more clips for your custom sign-language dataset.")
    parser.add_argument("--sentence", type=str, help="Sentence label directory, e.g. 'thank you'.")
    parser.add_argument("--count", type=int, default=1, help="Number of clips to record in this session.")
    parser.add_argument("--duration", type=float, default=5.0, help="Seconds per recording clip.")
    parser.add_argument("--fps", type=float, default=20.0, help="Saved video FPS.")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index to use.")
    parser.add_argument("--width", type=int, default=640, help="Output frame width.")
    parser.add_argument("--height", type=int, default=480, help="Output frame height.")
    parser.add_argument("--pause", type=float, default=1.0, help="Seconds to pause between clips.")
    return parser.parse_args()


def get_next_video_path(sentence_dir: Path) -> Path:
    existing = sorted(sentence_dir.glob("*.avi"))
    next_index = len(existing) + 1
    return sentence_dir / f"{next_index}.avi"


def record_single_clip(video_path: Path, camera_index: int, duration: float, fps: float, width: int, height: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Webcam index {camera_index} is not available.")

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Could not initialize video writer for {video_path}.")

    print(f"\n🎥 Recording started: {video_path.name}")
    print(f"⏳ Recording for {duration:.1f} seconds...")

    start = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Failed to read frames from the webcam.")

            elapsed = time.time() - start
            remaining = max(0.0, duration - elapsed)

            cv2.putText(
                frame,
                f"Recording... {remaining:.1f}s",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
            cv2.imshow("Recording", frame)
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

    print(f"✅ Saved: {video_path}")


def main() -> None:
    args = parse_args()
    sentence = (args.sentence or input("Enter sentence name: ")).strip()
    if not sentence:
        raise ValueError("Sentence name cannot be empty.")

    sentence_dir = Path("custom_dataset") / sentence
    sentence_dir.mkdir(parents=True, exist_ok=True)

    current_total = len(sorted(sentence_dir.glob("*.avi")))
    for clip_number in range(1, args.count + 1):
        video_path = sentence_dir / f"{current_total + clip_number}.avi"
        try:
            record_single_clip(video_path, args.camera_index, args.duration, args.fps, args.width, args.height)
        except KeyboardInterrupt:
            print("\n⏹️ Recording cancelled by user.")
            break
        except Exception as exc:
            print(f"\n❌ Failed to record clip {clip_number}: {exc}")
            break

        if clip_number < args.count and args.pause > 0:
            print(f"Pausing for {args.pause:.1f} seconds before next clip...")
            time.sleep(args.pause)

    final_total = len(sorted(sentence_dir.glob("*.avi")))
    print(f"\n📦 Dataset folder now has {final_total} video(s) for '{sentence}'.")


if __name__ == "__main__":
    main()