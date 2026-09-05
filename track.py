"""Run the HUGTrack association stage with the released parameters."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from util.run_ucmc import run_ucmc


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, default=ROOT / "outputs/detections")
    parser.add_argument(
        "--params", type=Path, default=ROOT / "tracker/geovesselmot-params.json"
    )
    parser.add_argument(
        "--camera-dir", type=Path, default=ROOT / "cam_para/geovesselmot/test"
    )
    parser.add_argument(
        "--viewpoint-dir", type=Path, default=ROOT / "dmc/geovesselmot/test"
    )
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/tracks")
    parser.add_argument("--run-name", default="hugtrack")
    parser.add_argument("--sequences", nargs="*", default=None)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--high-score", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    cli = parse_args()
    det_dir = cli.detections.expanduser().resolve()
    params_path = cli.params.expanduser().resolve()
    camera_dir = cli.camera_dir.expanduser().resolve()
    viewpoint_dir = cli.viewpoint_dir.expanduser().resolve()
    output_root = cli.output_root.expanduser().resolve()

    for path in (det_dir, camera_dir, viewpoint_dir):
        if not path.is_dir():
            raise NotADirectoryError(path)
    if not params_path.is_file():
        raise FileNotFoundError(params_path)

    with params_path.open(encoding="utf-8") as stream:
        released_params = json.load(stream)

    sequences = cli.sequences or sorted(path.stem for path in det_dir.glob("*.txt"))
    if not sequences:
        raise RuntimeError(f"no detection files found under {det_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    for sequence in sequences:
        if sequence not in released_params:
            raise KeyError(f"no released parameters for sequence: {sequence}")
        for required in (
            det_dir / f"{sequence}.txt",
            camera_dir / f"{sequence}.txt",
            viewpoint_dir / f"{sequence}.txt",
        ):
            if not required.is_file():
                raise FileNotFoundError(required)

        params = released_params[sequence]
        os.environ["SHAPE_W"] = str(params.get("shape_w", 0.0))
        run_args = argparse.Namespace(
            seq=sequence,
            fps=cli.fps,
            wx=params["wx"],
            wy=params["wy"],
            a=params["a"],
            cdt=params["cdt"],
            vmax=params["vmax"],
            conf_thresh=params["conf"],
            high_score=cli.high_score,
            hp=True,
            cmc=False,
            add_cam_noise=False,
            pose_delta=None,
            u_ratio=0.05,
            v_ratio=0.05,
            u_max=13,
            v_max=10,
        )
        run_ucmc(
            run_args,
            str(det_dir),
            str(camera_dir),
            str(viewpoint_dir),
            str(output_root),
            cli.run_name,
            "geovesselmot",
        )


if __name__ == "__main__":
    main()
