"""Compare saved snapshots side by side.

    python compare.py --latest 4                  the 4 newest snapshots
    python compare.py --latest 4 --panel 3        only panel 3 of each (e.g. FrED fixed T)
    python compare.py snapshots/a.png snapshots/b.png
    python compare.py --latest 4 --out figure.png  also save the comparison

Each snapshot shows its light setting and readings underneath. Press any key to close.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

import labkit


def snapshot_files(paths, latest):
    if paths:
        result = []
        for p in paths:
            path = Path(p)
            if not path.exists() and (labkit.SNAPSHOT_DIR / p).exists():
                path = labkit.SNAPSHOT_DIR / p
            result.append(path)
        return result
    if not labkit.SNAPSHOT_DIR.exists():
        return []
    files = sorted((p for p in labkit.SNAPSHOT_DIR.glob("*.png")
                    if "__" not in p.name and p.with_suffix(".json").exists()),
                   key=lambda p: p.stat().st_mtime)
    return files[-latest:]


def caption(info: dict, width: int) -> np.ndarray:
    lines = [f"{info.get('script', '')}   light: {info.get('light', '-')} %   {info.get('time', '')}"]
    lines += [f"{k}: {v}" for k, v in info.get("readings", {}).items()]
    bar = np.zeros((8 + 18 * len(lines), width, 3), np.uint8)
    for i, text in enumerate(lines):
        labkit.put_text(bar, text, (6, 18 + 18 * i), (120, 255, 120) if i else (255, 255, 255))
    return bar


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved snapshots side by side")
    parser.add_argument("files", nargs="*", help="snapshot .png files (default: newest ones)")
    parser.add_argument("--latest", type=int, default=4, help="how many of the newest snapshots")
    parser.add_argument("--panel", type=int, help="show only this panel number (0 = first)")
    parser.add_argument("--out", type=Path, help="also save the comparison to this file")
    parser.add_argument("--no-window", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    files = snapshot_files(args.files, args.latest)
    if not files:
        raise SystemExit(f"No snapshots found in {labkit.SNAPSHOT_DIR}. Press s in a lab window first.")
    tiles = []
    for path in files:
        info_path = path.with_suffix(".json")
        info = json.loads(info_path.read_text()) if info_path.exists() else {}
        image_path = path.with_name(f"{path.stem}__p{args.panel}.png") if args.panel is not None else path
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"skipping {image_path} (not found)")
            continue
        width = 480 if args.panel is not None or len(files) > 2 else 760
        img = cv2.resize(img, (width, int(img.shape[0] * width / img.shape[1])), interpolation=cv2.INTER_AREA)
        if args.panel is not None and info.get("panels"):
            labkit.put_text(img, info["panels"][args.panel], (6, 18))
        tiles.append(np.vstack([img, caption(info, width)]))
        print(f"{path.name}: light {info.get('light')}  {labkit.Lab._readings_text(info.get('readings', {}))}")
    if not tiles:
        raise SystemExit("Nothing to show (check --panel: not every exercise has that many panels).")
    height = max(t.shape[0] for t in tiles)
    tiles = [np.vstack([t, np.zeros((height - t.shape[0], t.shape[1], 3), np.uint8)]) for t in tiles]
    cols = 2 if len(tiles) == 4 else min(len(tiles), 3)
    rows = [tiles[i:i + cols] for i in range(0, len(tiles), cols)]
    rows[-1] += [np.zeros_like(tiles[0])] * (cols - len(rows[-1]))
    sheet = np.vstack([np.hstack(r) for r in rows])
    scale = min(1.0, 1800 / sheet.shape[1], 1000 / sheet.shape[0])
    if scale < 1:
        sheet = cv2.resize(sheet, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    if args.out:
        cv2.imwrite(str(args.out), sheet)
        print(f"saved {args.out}")
    if not args.no_window:
        cv2.imshow("compare snapshots", sheet)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
