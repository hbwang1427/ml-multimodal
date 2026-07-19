import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoImageProcessor, AutoTokenizer

from .config import Config
from .dataset import ImageTextDataset, make_collate_fn
from .model import FusionCLIPModel, clip_contrastive_loss


@torch.no_grad()
def recall_at_1(model, loader, device):
    model.eval()
    image_embeds, text_embeds = [], []
    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        image_embeds.append(model.encode_image(pixel_values))
        text_embeds.append(model.encode_text(input_ids, attention_mask))
    image_embeds = torch.cat(image_embeds)
    text_embeds = torch.cat(text_embeds)
    sims = image_embeds @ text_embeds.t()
    preds = sims.argmax(dim=1)
    targets = torch.arange(len(preds), device=preds.device)
    return (preds == targets).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    args = parser.parse_args()

    cfg = Config()
    for field in ("data_dir", "output_dir", "epochs", "batch_size", "lr"):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)

    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name)
    image_processor = AutoImageProcessor.from_pretrained(cfg.vision_model_name)

    data_dir = Path(cfg.data_dir)
    collate_fn = make_collate_fn(tokenizer, cfg.max_length)

    train_set = ImageTextDataset(
        data_dir / "train.jsonl", data_dir / "images", tokenizer, image_processor, cfg.max_length,
    )
    val_set = ImageTextDataset(
        data_dir / "val.jsonl", data_dir / "images", tokenizer, image_processor, cfg.max_length,
    )
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=False, collate_fn=collate_fn)

    model = FusionCLIPModel(
        cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
        freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text,
    ).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable_params)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable:,} / {n_total:,} ({100 * n_trainable / n_total:.1f}%)")

    optimizer = torch.optim.AdamW(trainable_params, lr=cfg.lr)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            logits_per_image, logits_per_text = model(pixel_values, input_ids, attention_mask)
            loss = clip_contrastive_loss(logits_per_image, logits_per_text)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        r1 = recall_at_1(model, val_loader, device)
        print(f"epoch {epoch + 1}/{cfg.epochs}  loss={avg_loss:.4f}  val_recall@1={r1:.3f}")

    ckpt_path = Path(cfg.output_dir) / "fusion_model.pt"
    torch.save({"model_state": model.state_dict(), "config": vars(cfg)}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
