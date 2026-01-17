from pathlib import Path

from scripts.video_to_frames import extract_video_frames
from scripts.face_extraction import extract_face_tracks
from scripts.window_dataset_builder import build_temporal_windows


DATA_ROOT = Path("data/videos")
OUT_ROOT = Path("processed")

TARGET_FPS = 10
WINDOW_SIZE = 16
STRIDE = 8


def main():
    OUT_ROOT.mkdir(exist_ok=True)

    for label in ["real", "fake"]:
        label_root = DATA_ROOT / label

        if not label_root.exists():
            print(f"⚠️ Skipping missing folder: {label_root}")
            continue

        # Recursively find all videos
        for video_path in label_root.rglob("*.mp4"):
            video_id = f"{label}_{video_path.stem}"
            out_dir = OUT_ROOT / video_id

            if out_dir.exists():
                print(f"⏩ Skipping already processed: {video_id}")
                continue

            print(f"▶ Processing: {video_path}")

            # Script 1: Frames
            extract_video_frames(
                video_path=str(video_path),
                output_dir=str(out_dir),
                target_fps=TARGET_FPS
            )

            # Script 2: Face detection & tracking
            extract_face_tracks(
                frames_dir=str(out_dir / "frames"),
                output_dir=str(out_dir)
            )

            # Script 3: Temporal windows
            build_temporal_windows(
                video_dir=str(out_dir),
                label=label,
                window_size=WINDOW_SIZE,
                stride=STRIDE
            )

    print("✅ Batch preprocessing complete.")


if __name__ == "__main__":
    main()
