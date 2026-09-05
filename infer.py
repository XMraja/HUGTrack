"""Run HUGTrack detection on GeoVesselMOT test sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/geovesselmot.pt")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, default=ROOT / "outputs/detections")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Detection confidence threshold.",
    )
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def strip_training_branches(model) -> None:
    for name in ("gld_aux", "kd_anchor"):
        model._modules.pop(name, None)
        model.__dict__.pop(name, None)


def main() -> None:
    args = parse_args()
    weights = args.weights.expanduser().resolve()
    data_root = args.data_root.expanduser().resolve()
    out_root = args.out_root.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not data_root.is_dir():
        raise NotADirectoryError(data_root)

    model = YOLO(str(weights), task="detect")
    strip_training_branches(model.model)
    out_root.mkdir(parents=True, exist_ok=True)

    sequences = sorted(path for path in data_root.iterdir() if path.is_dir())
    if not sequences:
        raise RuntimeError(f"no sequence directories found under {data_root}")

    for sequence in sequences:
        image_dir = sequence / "img1"
        images = sorted(
            path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not images:
            raise RuntimeError(f"no images found under {image_dir}")
        output_file = out_root / f"{sequence.name}.txt"
        with output_file.open("w", encoding="utf-8") as stream:
            for frame_id, image_path in enumerate(tqdm(images, desc=sequence.name), start=1):
                result = model.predict(
                    str(image_path),
                    imgsz=args.imgsz,
                    conf=args.conf,
                    device=args.device,
                    verbose=False,
                )[0]
                for detection_id, box in enumerate(result.boxes.cpu().numpy(), start=1):
                    x1, y1, x2, y2 = box.xyxy[0]
                    confidence = float(box.conf[0])
                    width, height = x2 - x1, y2 - y1
                    stream.write(
                        f"{frame_id},{detection_id},{x1:.1f},{y1:.1f},"
                        f"{width:.1f},{height:.1f},{confidence:.2f},-1,-1,-1\n"
                    )


if __name__ == "__main__":
    main()
