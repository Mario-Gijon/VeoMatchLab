#!/usr/bin/env python3

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_ROOT = Path(
    "experiments/tracking/reid_crop_debug_margin15"
)


def load_images(track_dir: Path):
    images = []

    for path in sorted(track_dir.glob("*.jpg")):
        if path.name == "contact_sheet.jpg":
            continue

        img = cv2.imread(str(path))
        if img is None:
            continue

        images.append((path, img))

    return images


def make_sheet(raw_id: int, root: Path, output: Path):
    track_dir = root / f"track_{raw_id:03d}"

    if not track_dir.exists():
        raise FileNotFoundError(
            f"No existe: {track_dir}"
        )

    rows = load_images(track_dir)

    if not rows:
        raise RuntimeError(
            f"No hay crops JPG en {track_dir}"
        )

    thumb_w = 180
    thumb_h = 360
    label_h = 45

    cells = []

    for path, img in rows:
        resized = cv2.resize(
            img,
            (thumb_w, thumb_h),
            interpolation=cv2.INTER_AREA,
        )

        cell = np.zeros(
            (
                thumb_h + label_h,
                thumb_w,
                3,
            ),
            dtype=np.uint8,
        )

        cell[:thumb_h] = resized

        # Intenta sacar el frame del nombre.
        text = path.stem

        cv2.putText(
            cell,
            text,
            (5, thumb_h + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

        cells.append(cell)

    columns = min(5, len(cells))
    rows_count = (
        len(cells) + columns - 1
    ) // columns

    sheet = np.zeros(
        (
            rows_count * (thumb_h + label_h),
            columns * thumb_w,
            3,
        ),
        dtype=np.uint8,
    )

    for i, cell in enumerate(cells):
        row = i // columns
        col = i % columns

        y = row * (thumb_h + label_h)
        x = col * thumb_w

        sheet[
            y:y + thumb_h + label_h,
            x:x + thumb_w,
        ] = cell

    # Título arriba.
    title_h = 55

    final = np.zeros(
        (
            sheet.shape[0] + title_h,
            sheet.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    final[title_h:] = sheet

    cv2.putText(
        final,
        f"RAW TRACK {raw_id}",
        (15, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv2.imwrite(
        str(output),
        final,
    )

    print(f"RAW {raw_id}")
    print(f"Crops: {len(rows)}")
    print(f"Imagen: {output.resolve()}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "raw_id",
        type=int,
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    output = args.output

    if output is None:
        output = Path(
            f"/tmp/raw_{args.raw_id:03d}.jpg"
        )

    make_sheet(
        args.raw_id,
        args.root,
        output,
    )


if __name__ == "__main__":
    main()