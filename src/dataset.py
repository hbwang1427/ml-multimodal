import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset


class ImageTextDataset(Dataset):
    """Reads a JSONL manifest of {"image": "relative/path.png", "text": "caption"}
    records plus an image root directory. Works with the synthetic sample
    dataset out of the box, and with any dataset that's been converted to
    this same manifest format (see data/generate_synthetic.py)."""

    def __init__(self, manifest_path, image_root, tokenizer, image_processor, max_length=32):
        self.image_root = Path(image_root)
        with open(manifest_path) as f:
            self.records = [json.loads(line) for line in f]
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = Image.open(self.image_root / rec["image"]).convert("RGB")
        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        return {"pixel_values": pixel_values, "text": rec["text"]}


def make_collate_fn(tokenizer, max_length=32):
    def collate(batch):
        pixel_values = torch.stack([b["pixel_values"] for b in batch])
        texts = [b["text"] for b in batch]
        tokenized = tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        return {
            "pixel_values": pixel_values,
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
    return collate
