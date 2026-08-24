# Class 2: MLOps Basics, Interview Prep, and Resume Building
**Duration:** 60 minutes
**Prerequisite:** Class 1 (repo setup: [ml-multimodal](https://github.com/hbwang1427/ml-multimodal))

## Agenda
| Time | Topic |
|------|-------|
| 0:00–0:10 | DVC + Git for `src/` with GCP BigQuery & Cloud Storage |
| 0:10–0:50 | ML / DL / Transformer / Multimodal interview questions + coding |
| 0:50–1:00 | Resume content for internship applications |

---

## Part 1 — Versioning `src/` with Git + DVC on GCP (10 min)

### 1.1 Why two tools?
- **Git** tracks code (`src/`, notebooks, configs) — small, text-based, diffable.
- **DVC** (Data Version Control) tracks large artifacts — datasets, model checkpoints, features — that shouldn't live in Git. DVC stores lightweight `.dvc` pointer files in Git, and pushes the actual data to a remote (here, a GCS bucket).

### 1.2 Minimal setup
```bash
# In the repo root (ml-multimodal/)
git init                       # if not already
pip install dvc[gs]            # gs = Google Cloud Storage support

dvc init
git add .dvc .dvcignore
git commit -m "Initialize DVC"

# Point DVC at a GCS bucket as the remote storage
dvc remote add -d gcs-remote gs://<your-bucket-name>/dvc-store
git add .dvc/config
git commit -m "Configure GCS remote for DVC"
```

### 1.3 Everyday workflow
```bash
dvc add data/train.csv         # start tracking a data file
git add data/train.csv.dvc data/.gitignore
git commit -m "Track train.csv with DVC"

dvc push                       # upload data to gs://bucket
git push                       # push code + pointer files

# Teammate / new machine
git pull
dvc pull                       # pulls actual data from GCS
```

### 1.4 Where BigQuery fits in
- Raw/structured data (labels, metadata, logs) often lives in **BigQuery**, not flat files.
- Typical pattern in `src/data/`: a script (`load_bq.py`) queries BigQuery, materializes a snapshot to `data/raw/*.parquet`, and **that snapshot** gets `dvc add`-ed — so every experiment is tied to a reproducible data version, not a live, mutable table.
```python
from google.cloud import bigquery
client = bigquery.Client(project="your-gcp-project")
df = client.query("SELECT * FROM dataset.table WHERE split='train'").to_dataframe()
df.to_parquet("data/raw/train_snapshot.parquet")
```
- Rule of thumb: **BigQuery = source of truth for structured data; GCS + DVC = frozen, versioned snapshots used by training code.**

### 1.5 Takeaway
- Git → `src/`, configs, `.dvc` pointers.
- DVC → data/model binaries, backed by a GCS bucket.
- BigQuery → queried upstream to produce the snapshots DVC tracks.
- `dvc.yaml` (not covered in depth today) lets you define full pipelines (`query → preprocess → train → eval`) as reproducible, cacheable stages — worth exploring after this course.

---

## Part 2 — Interview Questions & Coding, Internship Track (40 min)

*Goal: rapid-fire conceptual questions + a couple of short coding drills you should be able to do in 15–20 minutes each, whiteboard-style. These mirror what's commonly asked in ML/AI internship screens.*

### 2.1 ML Fundamentals (10 min)
Conceptual, expect 1–2 sentence answers:
- Bias–variance tradeoff — what happens as model complexity increases?
- Why does L2 regularization shrink weights smoothly, while L1 induces sparsity?
- Precision vs. recall vs. F1 — when would you optimize for one over another?
- What does k-fold cross-validation protect you against that a single train/val split doesn't?
- Explain gradient descent vs. stochastic gradient descent vs. mini-batch. Why do we use mini-batches in practice?
- What's the difference between generative and discriminative models? Give one example of each.
- How does a decision tree choose a split (Gini vs. entropy)? Why do random forests reduce variance over a single tree?

**Coding drill (~10 min):** Implement logistic regression from scratch (gradient descent, no sklearn) — forward pass, binary cross-entropy loss, and a manual gradient update loop over a toy 2D dataset. Interviewers care about: correct gradient math, vectorization, and reasonable stopping criteria.

### 2.2 Deep Learning Fundamentals, especially Transformers (20 min)

**Core DL (5 min):**
- Why do sigmoid/tanh activations cause vanishing gradients in deep nets, and how do ReLU/GELU help?
- What problem does batch normalization solve? How does it differ from layer normalization (and why do transformers use layer norm)?
- What's the purpose of dropout, and why is it disabled at inference time?
- Explain the vanishing/exploding gradient problem in vanilla RNNs and how LSTMs/GRUs mitigate it with gates.

**Transformers (15 min) — the emphasis:**
- Walk through **scaled dot-product attention**: `softmax(QK^T / sqrt(d_k)) V`. Why divide by `sqrt(d_k)`?
- What is **multi-head attention** buying you over a single attention head?
- Why do transformers need **positional encoding**? What's the difference between sinusoidal and learned positional embeddings?
- Encoder-only (BERT) vs. decoder-only (GPT) vs. encoder-decoder (T5) — when is each architecture appropriate?
- What is **causal masking** and why is it needed in decoder self-attention?
- Why is self-attention O(n²) in sequence length, and what's the practical consequence for long sequences? (mention: sparse/linear attention, sliding window as known mitigations — no need to derive)
- What do Q, K, V represent conceptually — where do they come from?

**Coding drill (~10 min):** Implement scaled dot-product attention in NumPy or PyTorch for a single head, given `Q`, `K`, `V` matrices — including the softmax and the causal mask option:
```python
import numpy as np

def attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = np.exp(scores - scores.max(axis=-1, keepdims=True))
    weights /= weights.sum(axis=-1, keepdims=True)
    return weights @ V
```

### 2.3 Multimodal Fundamentals (10 min)
- What is **contrastive learning**, and how does CLIP use it to align image and text embeddings in a shared space?
- Compare **fusion strategies**: early fusion (concatenate raw/low-level features), late fusion (combine model outputs/decisions), and cross-attention fusion (one modality attends to another). Trade-offs of each?
- How does a **Vision Transformer (ViT)** turn an image into a sequence a transformer can consume (patch embedding)?
- Why is modality alignment (e.g., image-text pairs) harder to get right than single-modality training — think about noisy pairs, differing granularity, and missing modalities at inference time.
- What's the difference between a dual-encoder model (CLIP-style, one tower per modality) and a fusion-encoder model (single transformer jointly processing both modalities, e.g. ALBEF/BLIP-style)?

**Discussion prompt (no coding, 3–5 min):** Given an internship project of "classify product listings using title text + product image," would you propose early, late, or cross-attention fusion, and why? Be ready to defend the tradeoff (latency, data size, interpretability).

---

## Part 3 — Resume Content for Internship Applications (10 min)

### 3.1 Structure (top to bottom, one page)
1. **Header** — name, email, GitHub/LinkedIn, portfolio link.
2. **Education** — school, degree, expected grad date, relevant coursework (ML, DL, statistics), GPA if strong.
3. **Skills** — grouped: Languages (Python, SQL...), ML/DL (PyTorch, scikit-learn, HuggingFace...), Tools/Cloud (Git, DVC, GCP/BigQuery, Docker...).
4. **Projects** — 2–4 projects, most relevant first. This is where a repo like `ml-multimodal` becomes resume material.
5. **Experience** (if any) — internships, research assistantships, TA roles.
6. **(Optional) Publications / Competitions** — Kaggle rank, papers, hackathons.

### 3.2 Turning this course's project into a resume bullet
Formula: **Action verb + what you built + techniques/tools + quantified outcome.**

Weak: *"Worked on a multimodal ML project using transformers."*

Stronger:
> "Built a multimodal classification pipeline (text + image) using a CLIP-style dual-encoder in PyTorch; versioned datasets with DVC on GCS and structured queries via BigQuery, improving reproducibility across 3 experiment iterations."

> "Implemented scaled dot-product and multi-head attention from scratch in NumPy to validate understanding of transformer internals; applied to a fine-tuned BERT baseline achieving X% F1 on [dataset]."

### 3.3 Tips specific to internship applications
- **Quantify wherever possible**: accuracy/F1 deltas, dataset size, latency, % improvement — even rough numbers beat vague claims.
- **Lead with impact, not tool names** — tools go in the Skills section; bullets should read as accomplishments.
- **Tailor to the job description** — mirror keywords (e.g., "PyTorch," "distributed training," "LLM," "computer vision") that appear in the posting, without keyword-stuffing.
- **Link your GitHub** prominently — for internships, a clean, well-documented repo (README, clear commits, this course's project included) often matters as much as the resume itself.
- **Keep it to one page** — cut anything that doesn't support an ML/DL narrative.
- **Practice explaining every bullet out loud** — anything on the resume is fair game in the interview; be ready to go deep on it (ties back to Part 2).

---

## Homework / Before Class 3
1. Push a `dvc add`-tracked dataset snapshot to your GCS remote and confirm `dvc pull` works from a clean clone.
2. Implement multi-head attention (extend the single-head snippet above) and test it on a toy sequence.
3. Draft 3 resume bullets for your `ml-multimodal` project using the formula in 3.2.
