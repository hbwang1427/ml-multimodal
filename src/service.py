"""Minimal FastAPI wrapper around FusionCLIPModel for SaaS-style
deployment. See Class 4 tutorial, Part 2.2 (mle-fundamentals-class-4-mlops-metrics.md).

The model, tokenizer, and image processor are all loaded ONCE at process
startup -- never per-request. That's the difference between a notebook
cell and a service: a request handler should only ever do inference, not
model loading.

Run locally:
    uvicorn src.service:app --host 0.0.0.0 --port 8080

Env vars:
    CHECKPOINT_PATH   path to a trained checkpoint (default outputs/fusion_model.pt)
"""
import io
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel
from transformers import AutoImageProcessor, AutoTokenizer

from .config import Config
from .model import FusionCLIPModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("retrieval-service")

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "outputs/fusion_model.pt")

# Holds the loaded model/tokenizer/processor for the life of the process.
# A real multi-tenant service would put per-tenant config here too.
state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("loading checkpoint from %s", CHECKPOINT_PATH)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = Config(**ckpt["config"])
    device = torch.device(cfg.device)

    model = FusionCLIPModel(
        cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
        freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    state["model"] = model
    state["cfg"] = cfg
    state["device"] = device
    state["tokenizer"] = AutoTokenizer.from_pretrained(cfg.text_model_name)
    state["image_processor"] = AutoImageProcessor.from_pretrained(cfg.vision_model_name)
    log.info("model loaded on %s -- ready to serve", device)
    yield
    state.clear()


app = FastAPI(title="ml-multimodal retrieval service", lifespan=lifespan)


@app.get("/health")
def health():
    """Cloud Run / Kubernetes readiness probe target."""
    return {"status": "ok", "model_loaded": "model" in state}


class EmbedTextRequest(BaseModel):
    text: str


@app.post("/embed_text")
def embed_text(req: EmbedTextRequest):
    if not req.text.strip():
        raise HTTPException(400, "text must be non-empty")

    start = time.perf_counter()
    try:
        cfg, model, device = state["cfg"], state["model"], state["device"]
        tokenized = state["tokenizer"](
            [req.text], padding=True, truncation=True,
            max_length=cfg.max_length, return_tensors="pt",
        )
        with torch.no_grad():
            embed = model.encode_text(
                tokenized["input_ids"].to(device),
                tokenized["attention_mask"].to(device),
            )
    except Exception:
        log.exception("text embedding failed")
        raise HTTPException(500, "inference error")

    latency_ms = (time.perf_counter() - start) * 1000
    log.info("embed_text latency_ms=%.1f", latency_ms)
    return {"embedding": embed.squeeze(0).tolist(), "latency_ms": latency_ms}


@app.post("/embed_image")
async def embed_image(file: UploadFile = File(...)):
    start = time.perf_counter()
    try:
        model, device = state["model"], state["device"]
        image = Image.open(io.BytesIO(await file.read())).convert("RGB")
        pixel_values = state["image_processor"](images=image, return_tensors="pt")["pixel_values"]
        with torch.no_grad():
            embed = model.encode_image(pixel_values.to(device))
    except Exception:
        log.exception("image embedding failed")
        raise HTTPException(500, "inference error")

    latency_ms = (time.perf_counter() - start) * 1000
    log.info("embed_image latency_ms=%.1f", latency_ms)
    return {"embedding": embed.squeeze(0).tolist(), "latency_ms": latency_ms}
