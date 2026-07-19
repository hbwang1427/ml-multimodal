from dataclasses import dataclass

import torch


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Config:
    # Foundation models pulled from the Hugging Face Hub.
    vision_model_name: str = "facebook/dinov2-small"
    text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Fusion / projection settings.
    embed_dim: int = 256
    freeze_vision: bool = True
    freeze_text: bool = True

    # Data & training.
    max_length: int = 32
    batch_size: int = 32
    epochs: int = 5
    lr: float = 1e-4
    seed: int = 42
    data_dir: str = "data/synthetic"
    output_dir: str = "outputs"
    device: str = _default_device()
