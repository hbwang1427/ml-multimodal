import os

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from transformers import GPT2Model, GPT2Tokenizer


class ImageEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.vit = timm.create_model('vit_small_patch16_224', pretrained=True, num_classes=0)
        self.proj = nn.Linear(self.vit.num_features, output_dim, bias=False)

    def forward(self, x):
        feat = self.vit(x)
        return self.proj(feat)


class TextEncoder(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = GPT2Model.from_pretrained('gpt2')
        self.proj = nn.Linear(self.model.config.n_embd, output_dim, bias=False)

    def forward(self, texts):
        device = next(self.model.parameters()).device
        inputs = self.tokenizer(texts, return_tensors='pt', padding=True, truncation=True).to(device)
        output = self.model(**inputs)

        last_hidden_state = output.last_hidden_state
        attention_mask = inputs['attention_mask']
        last_token_idx = attention_mask.sum(dim=1) - 1

        batch_idx = torch.arange(last_hidden_state.size(0), device=last_token_idx.device)
        sent = last_hidden_state[batch_idx, last_token_idx]
        return self.proj(sent)


class CLIP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.image_encoder = ImageEncoder(embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

    def forward(self, images, texts):
        img = self.image_encoder(images)
        txt = self.text_encoder(texts)

        img = F.normalize(img, dim=-1)
        txt = F.normalize(txt, dim=-1)

        scale = self.logit_scale.exp()
        logits = scale * (img @ txt.T)
        return logits


@torch.no_grad()
def encode_images(clip_model, images, batch_size=64):
    """Embed a (N, C, H, W) tensor of images in batches, L2-normalized."""
    device = next(clip_model.parameters()).device
    embeds = []
    for i in range(0, images.size(0), batch_size):
        batch = images[i:i + batch_size].to(device)
        feat = clip_model.image_encoder(batch)
        embeds.append(F.normalize(feat, dim=-1).cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def encode_texts(clip_model, texts, batch_size=64):
    """Embed a list of strings in batches, L2-normalized."""
    embeds = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        feat = clip_model.text_encoder(batch)
        embeds.append(F.normalize(feat, dim=-1).cpu())
    return torch.cat(embeds, dim=0)


def retrieve(query_embeds, gallery_embeds, top_k=5):
    """Rank `gallery_embeds` against `query_embeds` by cosine similarity.

    Both arguments must be L2-normalized embeddings in the same CLIP
    space (e.g. output of `encode_images`/`encode_texts`). Works for
    text->image, image->text, or image->image retrieval — whichever
    pair of embeddings you pass in.

    Returns (indices, scores), each of shape (num_queries, top_k).
    """
    sims = query_embeds @ gallery_embeds.T
    top_k = min(top_k, gallery_embeds.size(0))
    scores, indices = sims.topk(top_k, dim=-1)
    return indices, scores


@torch.no_grad()
def zero_shot_classify(clip_model, images, class_names, templates=None, batch_size=64):
    """Classify `images` against `class_names` with no task-specific training.

    `templates` are format strings applied to each class name (e.g.
    "a photo of a {}"); their embeddings are averaged per class for a
    more robust prototype. Defaults to the bare class name.

    Returns (preds, probs): predicted class index per image, and the
    full (N, num_classes) softmax distribution.
    """
    if templates is None:
        templates = ["{}"]

    device = next(clip_model.parameters()).device

    class_embeds = []
    for name in class_names:
        prompts = [t.format(name) for t in templates]
        embeds = F.normalize(clip_model.text_encoder(prompts), dim=-1).mean(dim=0)
        class_embeds.append(F.normalize(embeds, dim=0))
    class_embeds = torch.stack(class_embeds).to(device)  # (num_classes, D)

    preds, probs = [], []
    for i in range(0, images.size(0), batch_size):
        batch = images[i:i + batch_size].to(device)
        img_embeds = F.normalize(clip_model.image_encoder(batch), dim=-1)
        logits = clip_model.logit_scale.exp() * img_embeds @ class_embeds.T
        batch_probs = logits.softmax(dim=-1)
        preds.append(batch_probs.argmax(dim=-1).cpu())
        probs.append(batch_probs.cpu())

    return torch.cat(preds, dim=0), torch.cat(probs, dim=0)


def load_cifar10_dataset(batch_size, image_size=224, root='./cifar10', mean=None, std=None):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    train_dataset = CIFAR10(root=root, train=True, download=True, transform=transform)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    classes = train_dataset.classes
    return loader, classes


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")

    clip_model = CLIP(embed_dim=512).to(device)

    cfg = clip_model.image_encoder.vit.default_cfg
    mean = cfg['mean']  # from the ViT's pretraining normalization stats
    std = cfg['std']
    data_root = os.path.join(os.path.dirname(__file__), "cifar10")
    data_loader, classes = load_cifar10_dataset(batch_size=4, root=data_root, mean=mean, std=std)

    optimizer = torch.optim.AdamW(clip_model.parameters(), lr=1e-5)

    # training procedure
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        # turn labels into text descriptions
        texts = [classes[label.item()] for label in labels]

        # get the image-text affinity matrix
        logits = clip_model(images, texts)

        # i-th image should match the i-th text in the batch
        targets = torch.arange(logits.shape[0]).to(device)

        # symmetric contrastive loss (image->text and text->image)
        loss_i = nn.CrossEntropyLoss()(logits, targets)
        loss_t = nn.CrossEntropyLoss()(logits.T, targets)
        loss = (loss_i + loss_t) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Batch {i}: Loss = {loss.item():.4f}")

        if i >= 5:
            break

    clip_model.eval()

    # --- zero-shot classification demo ---
    templates = ["a photo of a {}", "a blurry photo of a {}"]
    preds, probs = zero_shot_classify(clip_model, images, classes, templates=templates)
    true_labels = [classes[label.item()] for label in labels]
    pred_labels = [classes[p.item()] for p in preds]
    print(f"True labels: {true_labels}")
    print(f"Predicted:   {pred_labels}")

    # --- text -> image retrieval demo ---
    image_embeds = encode_images(clip_model, images)
    query_embeds = encode_texts(clip_model, [f"a photo of a {classes[0]}"])
    top_k_idx, top_k_scores = retrieve(query_embeds, image_embeds, top_k=2)
    print(f"Top matches for 'a photo of a {classes[0]}': "
          f"indices={top_k_idx.tolist()} scores={top_k_scores.tolist()}")
