"""Evaluate HUGTrack outputs with the bundled TrackEval adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.eval import eval as run_trackeval


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gt-root", type=Path, required=True)
    parser.add_argument("--track-root", type=Path, default=ROOT / "outputs/tracks")
    parser.add_argument("--seqmap", type=Path, required=True)
    parser.add_argument("--run-name", default="hugtrack")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_trackeval(
        str(args.gt_root.expanduser().resolve()),
        str(args.track_root.expanduser().resolve()),
        str(args.seqmap.expanduser().resolve()),
        args.run_name,
        1,
        False,
    )
    names = ("HOTA", "IDF1", "MOTA", "AssA")
    print(json.dumps(dict(zip(names, map(float, metrics))), indent=2))


if __name__ == "__main__":
    main()
