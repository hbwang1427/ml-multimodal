"""Export FusionCLIPModel's vision and/or text tower (frozen backbone +
trained projection head + L2-normalize) to ONNX for on-device deployment.
See Class 4 tutorial, Part 2.3 (mle-fundamentals-class-4-mlops-metrics.md).

Each tower is exported as a single graph so a mobile/edge runtime only
has to run ONE model per modality -- no separate backbone/head loading.

Usage:
    python -m src.export_onnx --tower vision
    python -m src.export_onnx --tower text
    python -m src.export_onnx --tower both --checkpoint outputs/fusion_model.pt
"""
import argparse
import inspect
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .model import FusionCLIPModel

# torch>=2.5 defaults torch.onnx.export to its newer torch.export-based
# exporter, which needs the optional `onnxscript` package and (as of this
# writing) trips over this model's dynamic text-sequence axis. Force the
# older, more battle-tested TorchScript-based exporter when it's available;
# on torch<2.5 (still supported per requirements.txt) there's no such
# kwarg, and that legacy path is already the only exporter, so just omit it.
_ONNX_EXPORT_KWARGS = (
    {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
)


class VisionTower(nn.Module):
    """model.encode_image as a single traceable nn.Module graph."""

    def __init__(self, model: FusionCLIPModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.encode_image(pixel_values)


class TextTower(nn.Module):
    """model.encode_text as a single traceable nn.Module graph."""

    def __init__(self, model: FusionCLIPModel):
        super().__init__()
        self.model = model

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model.encode_text(input_ids, attention_mask)


def load_model(checkpoint: str) -> tuple[FusionCLIPModel, Config]:
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = Config(**ckpt["config"])
    model = FusionCLIPModel(
        cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
        freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text,
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


def _validate(torch_module: nn.Module, onnx_path: Path, inputs: dict[str, torch.Tensor]):
    """Confirm ONNX Runtime's output matches the PyTorch forward pass
    numerically -- never ship an export you haven't checked this way."""
    import onnxruntime as ort

    with torch.no_grad():
        torch_out = torch_module(*inputs.values()).numpy()

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {k: v.numpy() for k, v in inputs.items()})[0]

    max_diff = float(np.abs(torch_out - onnx_out).max())
    print(f"  [{onnx_path.name}] max abs diff PyTorch vs ONNX Runtime: {max_diff:.2e}")
    assert max_diff < 1e-3, f"{onnx_path.name} diverges from PyTorch -- do not ship this export"


def export_vision(model: FusionCLIPModel, out_path: Path, image_size: int = 224):
    tower = VisionTower(model).eval()
    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        tower, dummy, out_path.as_posix(),
        input_names=["pixel_values"], output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17, **_ONNX_EXPORT_KWARGS,
    )
    print(f"exported vision tower -> {out_path}")
    _validate(tower, out_path, {"pixel_values": dummy})


def export_text(model: FusionCLIPModel, out_path: Path, cfg: Config):
    tower = TextTower(model).eval()
    dummy_ids = torch.randint(0, 1000, (1, cfg.max_length))
    dummy_mask = torch.ones(1, cfg.max_length, dtype=torch.long)
    torch.onnx.export(
        tower, (dummy_ids, dummy_mask), out_path.as_posix(),
        input_names=["input_ids", "attention_mask"], output_names=["embedding"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "embedding": {0: "batch"},
        },
        opset_version=17, **_ONNX_EXPORT_KWARGS,
    )
    print(f"exported text tower -> {out_path}")
    _validate(tower, out_path, {"input_ids": dummy_ids, "attention_mask": dummy_mask})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/fusion_model.pt")
    parser.add_argument("--tower", choices=["vision", "text", "both"], default="both")
    parser.add_argument("--out_dir", default="outputs/onnx")
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model, cfg = load_model(args.checkpoint)

    if args.tower in ("vision", "both"):
        export_vision(model, out_dir / "vision_tower.onnx", args.image_size)
    if args.tower in ("text", "both"):
        export_text(model, out_dir / "text_tower.onnx", cfg)

    print(
        "\nNext step (NVIDIA GPU/Jetson target): "
        "bash deploy/export_tensorrt.sh <onnx_path> <engine_path>"
    )


if __name__ == "__main__":
    main()
