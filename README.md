# ml-multimodal: vision-language fusion (light training)

A CLIP-style dual-encoder that pairs a frozen Hugging Face **vision
foundation model** with a frozen Hugging Face **text encoder**, and only
trains lightweight **projection heads** on top to align the two into a
shared embedding space via contrastive (InfoNCE) loss. This keeps training
cheap — no backbone fine-tuning required.

## Framework

```
image ──► VisionEncoder (frozen, e.g. facebook/dinov2-small)
              │ CLS token (hidden_size)
              ▼
          ProjectionHead (trainable) ──► L2-normalized embedding ┐
                                                                   ├─► InfoNCE
          ProjectionHead (trainable) ──► L2-normalized embedding ┘   (contrastive)
              ▲
              │ mean-pooled tokens (hidden_size)
text ──► TextEncoder (frozen, e.g. sentence-transformers/all-MiniLM-L6-v2)
```

- **Vision foundation model**: any HF `AutoModel` with a ViT-style
  `last_hidden_state` output (DINOv2, ViT, BEiT, ...). Default:
  `facebook/dinov2-small`.
- **Text encoder**: any HF `AutoModel` (BERT/DistilBERT/MiniLM-style).
  Default: `sentence-transformers/all-MiniLM-L6-v2`.
- **Fusion**: both backbones stay frozen (`requires_grad=False`); only the
  two small `ProjectionHead` MLPs (+ a learned temperature) are trained.
  Swap `freeze_vision`/`freeze_text` in `src/config.py` to unfreeze and
  fine-tune the backbones too, if you have more compute.
- An optional `CrossAttentionFusion` module is included in `src/model.py`
  for tasks needing a single joint representation (VQA, image-text
  matching classification) instead of a dual retrieval embedding — not
  wired into the default training loop, but ready to plug in.

## Project layout

```
data/
  generate_synthetic.py   # builds the offline sample dataset
  synthetic/               # generated: images/, train.jsonl, val.jsonl
src/
  config.py                # all hyperparameters / model names
  model.py                 # VisionEncoder, TextEncoder, FusionCLIPModel
  dataset.py                # JSONL manifest -> (pixel_values, text) dataset
  train.py                  # contrastive training loop
  infer.py                  # text -> image retrieval demo
requirements.txt
```

## Sample dataset

`data/generate_synthetic.py` procedurally generates a tiny, fully offline
image-caption dataset (colored shapes on colored backgrounds, e.g. *"a red
circle on a white background"*) — 300 train / 60 val pairs — so the whole
pipeline can be smoke-tested with zero downloads beyond the two HF models.

Manifest format (`train.jsonl` / `val.jsonl`), one JSON object per line:

```json
{"image": "img_00000.png", "text": "a red circle on a white background"}
```

To use a **real** dataset instead, load anything from the Hugging Face Hub
with `datasets.load_dataset(...)` (e.g. `nlphuji/flickr30k`,
`HuggingFaceM4/COCO`) and write it out in this same `{"image", "text"}`
JSONL + image-folder format — no other code changes needed.

## Running it

```bash
pip install -r requirements.txt

python data/generate_synthetic.py          # builds data/synthetic/

python -m src.train                         # trains projection heads
python -m src.infer --query "a blue square on a black background"
```

`train.py` prints the trainable-vs-total parameter count each run so you
can see how light the training actually is (typically <5% of total
params), plus per-epoch loss and validation recall@1. Override defaults
via flags, e.g.:

```bash
python -m src.train --epochs 10 --batch_size 64 --lr 3e-4
```

## Extending

- **Different backbones**: change `vision_model_name` / `text_model_name`
  in `src/config.py` to any compatible HF model id (e.g.
  `google/vit-base-patch16-224-in21k`, `distilbert-base-uncased`).
- **Downstream classification (VQA / ITM)**: use `CrossAttentionFusion` in
  `src/model.py` with `pooled=False` on both encoders to get full token
  sequences, then attach a classification head on the pooled fusion output.
- **Unfreezing backbones**: set `freeze_vision=False` / `freeze_text=False`
  in `Config` for full fine-tuning once the light-training baseline works.
