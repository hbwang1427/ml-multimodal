"""Generates a tiny, fully offline image-caption dataset of colored shapes.

This exists purely to smoke-test the vision-language fusion pipeline without
needing to download a real dataset. Swap it out for a real Hugging Face
dataset (e.g. via `datasets.load_dataset(...)`) once the pipeline is
verified -- just write out the same {"image", "text"} JSONL manifest format.
"""
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red": (220, 60, 60),
    "green": (60, 170, 90),
    "blue": (60, 100, 220),
    "yellow": (230, 200, 60),
    "purple": (150, 70, 190),
    "orange": (230, 140, 50),
}
BACKGROUNDS = {
    "white": (250, 250, 250),
    "black": (20, 20, 20),
    "gray": (150, 150, 150),
}
SIZE = 128


def draw_shape(draw, shape, color, box):
    if shape == "circle":
        draw.ellipse(box, fill=color)
    elif shape == "square":
        draw.rectangle(box, fill=color)
    elif shape == "triangle":
        x0, y0, x1, y1 = box
        draw.polygon([((x0 + x1) / 2, y0), (x0, y1), (x1, y1)], fill=color)


def make_example(rng):
    shape = rng.choice(SHAPES)
    color_name, color_rgb = rng.choice(list(COLORS.items()))
    bg_name, bg_rgb = rng.choice(list(BACKGROUNDS.items()))
    while bg_rgb == color_rgb:
        bg_name, bg_rgb = rng.choice(list(BACKGROUNDS.items()))

    img = Image.new("RGB", (SIZE, SIZE), bg_rgb)
    draw = ImageDraw.Draw(img)
    margin = rng.randint(20, 40)
    box = (margin, margin, SIZE - margin, SIZE - margin)
    draw_shape(draw, shape, color_rgb, box)

    caption = f"a {color_name} {shape} on a {bg_name} background"
    return img, caption


def build_split(images_dir, n, rng, start_idx):
    records = []
    for i in range(n):
        img, caption = make_example(rng)
        filename = f"img_{start_idx + i:05d}.png"
        img.save(images_dir / filename)
        records.append({"image": filename, "text": caption})
    return records


def main():
    out_dir = Path(__file__).parent / "synthetic"
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    train_records = build_split(images_dir, n=300, rng=rng, start_idx=0)
    val_records = build_split(images_dir, n=60, rng=rng, start_idx=300)

    with open(out_dir / "train.jsonl", "w") as f:
        for rec in train_records:
            f.write(json.dumps(rec) + "\n")
    with open(out_dir / "val.jsonl", "w") as f:
        for rec in val_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(train_records)} train / {len(val_records)} val examples to {out_dir}")


if __name__ == "__main__":
    main()
