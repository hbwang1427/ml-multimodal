# Machine Learning Engineer Fundamentals
## Class 4: MLOps & Inference Metrics — SaaS vs. On-Device vs. LLM/VLM

> Builds on Class 2 (Git/DVC/BigQuery/GCS tooling) and Class 3 (production
> data pipelines). This class turns from "how do we version and pipeline
> data" to "how do we deploy a model, and how do we know it's actually
> good once it's live" — for three different targets: a multi-tenant
> cloud API, an on-device export, and an LLM/VLM serving stack.

**Duration:** 90 minutes
**Prerequisite:** Class 2 (Git/DVC/GCP setup), Class 3 (data pipelines)

---

## Table of Contents
1. [Recap & Class 4 Overview](#1-recap--class-4-overview)
2. [Part 1 — Review: GCP & ML Tooling Setup (15 min)](#2-part-1--review-gcp--ml-tooling-setup-15-min)
3. [Part 2 — Deployment Pipelines & Inference Metrics by Industry (75 min)](#3-part-2--deployment-pipelines--inference-metrics-by-industry-75-min)
4. [Course Project: Instrument This Repo](#4-course-project-instrument-this-repo)
5. [Metrics Cheat Sheet](#5-metrics-cheat-sheet)
6. [Course Discussion](#6-course-discussion)
7. [Homework / Next Steps](#7-homework--next-steps)

---

## 1. Recap & Class 4 Overview

### Where We Left Off

Class 2 got code and data under version control (Git + DVC + GCS +
BigQuery). Class 3 turned that into repeatable data pipelines for
structured and unstructured (image) data. Both classes assumed the
*destination* of all this work was "a trained model." Class 4 is about
what happens after training: how the model actually gets **deployed**,
and how you measure whether it's ready to ship — because both the
deployment pipeline and the metrics that matter look different depending
on where the model runs: a cloud API serving thousands of tenants, a
phone with no network connection, or an LLM/VLM generating tokens one at
a time.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     WHY METRICS ARE THEIR OWN CLASS                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  "95% accuracy" tells you almost nothing about whether a model is          │
│  production-ready. A model can be accurate and still be un-shippable:      │
│  too slow under load, too expensive per request, too big for the           │
│  device, or too slow to generate the next token.                          │
│                                                                             │
│   Research Question                  │  Production Question                │
│   ──────────────────                 │  ────────────────────               │
│   • Is the model accurate?           │  • Is it accurate AND fast enough   │
│                                       │    AND cheap enough for where and   │
│                                       │    how it's deployed?               │
│   • Does it work on the test set?    │  • Does it work at p99 latency      │
│                                       │    under peak concurrent load?      │
│   • Can I reproduce this result?     │  • Can I roll this back in 2        │
│                                       │    minutes if it regresses?         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Class 4 Roadmap

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CLASS 4 SESSION FLOW (90 min)                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  0:00 ─ 0:15   Part 1: Review — GCP & ML tooling setup                     │
│  0:15 ─ 0:20   Part 2.1: A common metrics vocabulary                       │
│  0:20 ─ 0:40   Part 2.2: SaaS — cloud deployment pipeline + metrics        │
│  0:40 ─ 1:00   Part 2.3: On-device — ONNX/TensorRT pipeline + metrics      │
│  1:00 ─ 1:15   Part 2.4: LLM/VLM/multimodal — pipeline + metrics           │
│  1:15 ─ 1:20   Part 2.5: Side-by-side comparison                          │
│  1:20 ─ 1:30   Project kickoff + Q&A                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Learning Objectives

| Objective | You Will Be Able To |
|-----------|---------------------|
| **GCP/tooling review** | Verify your GCP project, `gcloud`, GCS, BigQuery, and DVC setup actually works end to end |
| **Metrics vocabulary** | Define latency (p50/p95/p99), throughput, and concurrency, and relate them with Little's Law |
| **SaaS deployment** | Describe a typical cloud deployment pipeline (build → container → serve → scale → monitor) and the metrics it's judged on |
| **On-device deployment** | Describe the ONNX/TensorRT (and mobile) export pipeline and the metrics it's judged on |
| **LLM/VLM deployment** | Describe a token-generation serving pipeline (continuous batching, KV cache) and its distinct metrics (TTFT, TPOT) |
| **Project** | Deploy this repo's retrieval model as a cloud service AND export it toward an on-device path, and measure both |

---

## 2. Part 1 — Review: GCP & ML Tooling Setup (15 min)

*Goal: before talking about deployment and production metrics, confirm the*
*foundation from Class 2/3 actually works on every machine. A metrics*
*discussion is useless if half the class can't run inference at all.*

### 2.1 The GCP + Tooling Stack So Far

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TOOLING STACK — WHAT EACH PIECE IS FOR                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Git ───────────── code, configs, `.dvc` pointers, PRs, CI triggers       │
│   DVC ───────────── large datasets / model checkpoints, GCS-backed         │
│   GCS ───────────── DVC remote storage + raw image/object storage          │
│   BigQuery ──────── structured data source-of-truth, queried into          │
│                      snapshots that DVC then versions                      │
│   gcloud CLI ────── auth, project/config selection, resource management    │
│   Artifact Registry ─ where we'll push Docker images (Part 2.2)            │
│   Cloud Run / Vertex AI ─ where SaaS inference metrics (Part 2.2) get       │
│                      measured once a model is actually deployed            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Environment Doctor — Run This Before Class

A single pass to confirm nothing has silently broken since Class 2/3
(expired auth tokens and stale `gcloud` configs are the #1 cause of "it
worked last week"):

```bash
# --- Identity & project ---
gcloud auth list                        # is the right account ACTIVE?
gcloud config list                      # correct project/region set?
gcloud projects describe $(gcloud config get-value project)

# --- Cloud Storage (GCS) ---
gsutil ls gs://<your-bucket-name>/      # can you list the bucket?
echo "ping" > /tmp/ping.txt && gsutil cp /tmp/ping.txt gs://<your-bucket-name>/ping.txt
gsutil rm gs://<your-bucket-name>/ping.txt   # round-trip write + delete

# --- BigQuery ---
bq query --use_legacy_sql=false 'SELECT 1 AS ok'

# --- DVC ---
dvc doctor                              # environment sanity check
dvc remote list                         # is the GCS remote still configured?
dvc pull                                # can you actually pull tracked data?

# --- Git ---
git remote -v
git log --oneline -5
git status

# --- Python / repo ---
python -c "import torch, transformers; print('torch', torch.__version__)"
python -m src.infer --query "a red circle on a white background"  # smoke test
```

### 2.3 Common Failure Modes

| Symptom | Likely Cause | Fix |
|---|---|---|
| `gcloud auth` shows no active account | Token expired | `gcloud auth login` |
| `gsutil ls` → `AccessDeniedException: 403` | Wrong project selected, or IAM role missing | `gcloud config set project <id>`; check IAM |
| `dvc pull` hangs or errors | Remote misconfigured or credentials not set | `dvc remote list -v`; re-run `gcloud auth application-default login` |
| `bq query` → permission error | BigQuery Data Viewer role missing on dataset | Ask project owner to grant role |
| `python -m src.infer` fails on import | Fresh clone, deps not installed | `pip install -r requirements.txt` |
| DVC pull succeeds but files are 0 bytes | Pulled pointer without actual GCS object (upload never finished) | Re-run `dvc push` from the machine that has the data, then `dvc pull` |

### 2.4 Review Checkpoint

```
BEFORE MOVING TO PART 2, EVERYONE SHOULD BE ABLE TO:
═══════════════════════════════════════════════════════════════
☐ Authenticate to the right GCP project via gcloud
☐ Read/write to the shared GCS bucket
☐ Run a BigQuery query against the course dataset
☐ dvc pull the tracked dataset/model snapshot successfully
☐ Run `python -m src.infer` locally end to end
```

If any box is unchecked, pair up with someone whose setup works while
they debug — this is exactly the kind of "works on my machine" gap that
Class 2's tooling exists to prevent, and it will resurface in Part 2's
deployment exercises if it isn't fixed now.

---

## 3. Part 2 — Deployment Pipelines & Inference Metrics by Industry (75 min)

### 3.1 A Common Metrics Vocabulary (5 min)

Before splitting into SaaS vs. on-device vs. LLM/VLM, everyone needs the
same base vocabulary — the same words get used differently across teams
otherwise.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     CORE INFERENCE METRICS VOCABULARY                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LATENCY       Time for one request to get a response. Always report as   │
│                a distribution, never an average:                          │
│                  p50 (median)  — the typical user's experience            │
│                  p95           — a slow-but-not-rare experience           │
│                  p99 / p99.9   — tail latency; what SLAs are built on      │
│                                                                             │
│  THROUGHPUT    Requests (or tokens) handled per unit time. A system can    │
│                have low latency AND low throughput (fast but can't        │
│                handle volume) — they are independent axes.                 │
│                                                                             │
│  CONCURRENCY   Number of requests being processed at the same instant.    │
│                Related to latency & throughput by Little's Law:           │
│                                                                             │
│                        L = λ · W                                          │
│                        (concurrency = arrival rate × avg latency)         │
│                                                                             │
│                e.g. 200 req/s arriving, 150ms avg latency                 │
│                     → ~30 requests in flight at any instant               │
│                                                                             │
│  COLD START    Extra latency the FIRST request pays (model load, JIT      │
│                compile, container spin-up) vs. WARM (steady-state) calls  │
│                                                                             │
│  UTILIZATION   % of available compute (GPU/CPU/NPU) actually doing work — │
│                low utilization at high cost is a red flag; too-high       │
│                utilization risks latency spikes under bursty traffic      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

Every one of the three deep dives below follows the same shape:
**deployment pipeline first, then the metrics that pipeline is judged on.**
The metrics only make sense once you know what's actually running.

---

### 3.2 SaaS: Cloud Deployment Pipeline → Metrics (20 min)

#### 3.2.1 Typical Cloud Deployment Pipeline

A SaaS inference service isn't "run `infer.py` on a bigger machine" — it's
a pipeline with distinct build-time and run-time stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  TYPICAL SAAS/CLOUD DEPLOYMENT PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BUILD TIME                                                                │
│  ───────────                                                               │
│  1. Package     Trained checkpoint + a serving handler                     │
│                  (src/infer.py logic wrapped for a request/response API)   │
│  2. Containerize Dockerfile: pinned deps, CUDA base image if GPU-serving,  │
│                  model weights baked in or pulled from GCS at startup      │
│  3. CI          Build image, run unit + smoke tests, push to               │
│                  Artifact Registry (tagged by commit SHA)                  │
│                                                                             │
│  DEPLOY TIME                                                               │
│  ────────────                                                              │
│  4. Serve       Deploy the image behind a managed runtime:                 │
│                    • Cloud Run (simplest — HTTP container, autoscale       │
│                      0→N, optional GPU)                                    │
│                    • GKE + a model server (Triton/TorchServe/BentoML) —    │
│                      more control, needed for custom batching              │
│                    • Vertex AI Endpoints — managed model serving,          │
│                      built-in traffic splitting for canaries               │
│  5. Front        API Gateway / Load Balancer: auth (API keys/OAuth),       │
│                  per-tenant rate limiting, request routing                 │
│  6. Scale        Autoscaling policy: min/max replicas, target concurrency  │
│                  per instance, scale-to-zero for low-traffic tenants       │
│  7. Observe      Cloud Monitoring + Logging: latency/error dashboards,     │
│                  alerting on SLO burn rate                                 │
│  8. Roll out     Canary or blue/green: new revision gets 5% of traffic,    │
│                  promoted to 100% only if metrics hold                     │
│                                                                             │
│  Client ──▶ LB/Gateway ──▶ [Revision N-1: 95%] ──▶ Model Server ──▶ (logs, │
│                        └──▶ [Revision N (canary): 5%] ──▶ Model Server     │  metrics)
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why the pipeline matters before the metrics:** p99 latency measured on
a single always-warm replica means nothing once real traffic hits
autoscaled, cold-starting, canary-split infrastructure. The metrics in
3.2.2 are only meaningful in the context of *this* pipeline.

#### 3.2.2 SaaS Inference Metrics

| Metric | What It Captures | Example SLO |
|---|---|---|
| **p95 / p99 latency** | Tail user experience, not the average | p99 < 300ms end-to-end |
| **Throughput (QPS)** | Peak sustained requests the service can absorb | 2,000 QPS at peak traffic |
| **Concurrency headroom** | How much burst above steady-state before degrading | 3x peak before p99 breaches SLO |
| **Autoscaling lag** | Time from traffic spike to new capacity online | New replica ready < 60s |
| **Batch fill rate** | % of max batch size actually used (GPU efficiency) | >70% average fill |
| **Cost per 1K inferences** | Unit economics — does pricing cover compute? | Tracked per tenant tier |
| **Error / timeout rate** | Requests that failed or exceeded deadline | <0.1% |
| **Availability (uptime)** | % of time the service met its SLA | 99.9% (≈43 min downtime/mo) |
| **Noisy-neighbor isolation** | One tenant's burst doesn't degrade others' latency | Per-tenant rate limits enforced |

#### 3.2.3 Worked Example: Deploying the Retrieval API to Cloud Run

```dockerfile
# Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
ENV MODEL_SNAPSHOT_URI=gs://<your-bucket-name>/models/fusion_clip_v3
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
# Build & push
gcloud builds submit --tag gcr.io/$(gcloud config get-value project)/retrieval-api

# Deploy with autoscaling (min 1 to avoid cold start on every canary test,
# max 20, allow 40 concurrent requests per instance)
gcloud run deploy retrieval-api \
  --image gcr.io/$(gcloud config get-value project)/retrieval-api \
  --min-instances=1 --max-instances=20 --concurrency=40 \
  --allow-unauthenticated

# Load test at increasing concurrency (see Section 4's project steps)
hey -z 60s -c 50 -m POST -d '{"text":"a red circle"}' \
  https://retrieval-api-xxxxx.run.app/embed
```

**Example results** (illustrative — your numbers will differ):

| Concurrency | p50 | p95 | p99 | QPS | Notes |
|---|---|---|---|---|---|
| 1 | 40ms | 55ms | 70ms | 20 | single-instance, warm |
| 10 | 45ms | 90ms | 140ms | 180 | still within one instance |
| 50 | 60ms | 210ms | 480ms | 620 | autoscaler adds instances mid-test |
| 200 | 65ms | 240ms | 510ms | 2,100 | steady-state after scale-out |

The p99 spike between concurrency 10→50 is the autoscaling lag from
3.2.1 step 6 showing up directly in the numbers — exactly why the
pipeline has to be understood before the metric is interpreted.

---

### 3.3 On-Device: ONNX/TensorRT Deployment Pipeline → Metrics (20 min)

#### 3.3.1 Typical On-Device Deployment Pipeline

On-device deployment is a graph-conversion pipeline, not a "deploy to a
server" pipeline — the model itself is transformed at each stage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              TYPICAL ON-DEVICE DEPLOYMENT PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. TRAIN (cloud)                                                          │
│     PyTorch checkpoint, FP32, training-only ops still in the graph         │
│                          │                                                 │
│                          ▼                                                 │
│  2. SIMPLIFY             Fuse batchnorm into conv, strip dropout/training  │
│     torch.onnx.export()  ops, set the model to eval() mode                │
│                          │                                                 │
│                          ▼                                                 │
│  3. EXPORT TO ONNX       A framework-neutral graph (opset version pinned); │
│     onnx-simplifier      validate: onnxruntime output ≈ torch output       │
│                          (np.allclose within tolerance)                    │
│                          │                                                 │
│              ┌───────────┴────────────┐                                   │
│              ▼                        ▼                                   │
│  4a. NVIDIA GPU / Jetson    4b. MOBILE (phone/tablet)                      │
│      trtexec / TensorRT         coremltools → CoreML (iOS, uses ANE)       │
│      builder API                TFLite converter → TFLite (Android,       │
│      • choose precision:          NNAPI/GPU delegate)                     │
│        FP16 or INT8             • quantize: dynamic, static (calibration  │
│      • INT8 needs a               dataset), or QAT                        │
│        calibration dataset                                                 │
│      • build a serialized                                                 │
│        .engine file                                                       │
│              │                        │                                   │
│              ▼                        ▼                                   │
│  5. BENCHMARK ON TARGET HARDWARE — never trust desktop/dev-machine numbers │
│              │                                                             │
│              ▼                                                             │
│  6. PACKAGE & SHIP — bundle into the app/device image; define an OTA       │
│     update path for future model refreshes, with rollback on failure      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why TensorRT specifically:** ONNX Runtime alone already runs on most
hardware, but **TensorRT** takes the extra step of compiling the ONNX
graph into a hardware-specific, kernel-fused, precision-reduced engine
for NVIDIA GPUs/Jetson boards — typically the biggest single latency win
in the whole pipeline, at the cost of a build step that has to be
re-run per target GPU architecture.

#### 3.3.2 On-Device Inference Metrics

| Metric | What It Captures | Example Target |
|---|---|---|
| **Model size (disk)** | App download/update size budget | < 20 MB for OTA-friendly updates |
| **Peak memory (RSS)** | Won't get OOM-killed by the OS | < 150 MB on lowest-spec target device |
| **Latency (warm)** | Per-inference time once loaded | < 50ms for a "real-time" UX feel |
| **Cold-start latency** | Time to first result (model/engine load) | TensorRT engine load < 500ms |
| **Engine build time** | One-time cost per target GPU (TensorRT-specific) | Tracked at CI/release time, not runtime |
| **Energy per inference** | Battery impact, not just speed | Bounded mJ/inference budget per feature |
| **Sustained battery drain** | Long-session usability | < X%/hour during continuous use |
| **Thermal throttling onset** | When sustained use degrades speed | No throttling within a 10-min session |
| **Accuracy delta (post-quant)** | Cost of FP16/INT8 compression | < 1-2 pt drop vs. FP32 baseline |
| **Offline success rate** | Works with zero network dependency | 100% — no silent network fallback |

#### 3.3.3 Worked Example: ONNX → TensorRT for the Vision Encoder

```python
# export_onnx.py — export this repo's VisionEncoder + ProjectionHead
import torch
from src.model import VisionEncoder, ProjectionHead
from src.config import Config

config = Config()
vision = VisionEncoder(config).eval()
head = ProjectionHead(config.vision_hidden_size, config.embed_dim).eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(
    torch.nn.Sequential(vision, head), dummy, "vision_tower.onnx",
    input_names=["pixel_values"], output_names=["embedding"],
    dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
    opset_version=17,
)
```

```bash
# Validate ONNX matches PyTorch numerically
python -c "
import onnxruntime as ort, torch, numpy as np
sess = ort.InferenceSession('vision_tower.onnx')
x = torch.randn(1, 3, 224, 224)
onnx_out = sess.run(None, {'pixel_values': x.numpy()})[0]
# compare against the torch forward pass, expect np.allclose(..., atol=1e-3)
"

# Build a TensorRT FP16 engine and benchmark on the TARGET GPU
trtexec --onnx=vision_tower.onnx --saveEngine=vision_tower_fp16.engine --fp16
trtexec --loadEngine=vision_tower_fp16.engine --avgRuns=100 --warmUp=20
```

**Example results** (illustrative — always re-measure on your actual
target hardware, e.g. a Jetson Orin, not a desktop dev GPU):

| Stage | Size | Latency (batch=1) | Notes |
|---|---|---|---|
| PyTorch FP32 (CPU) | 88 MB | 210ms | baseline, not shippable |
| ONNX Runtime FP32 (CPU) | 88 MB | 165ms | graph opts, still CPU-only |
| TensorRT FP16 (GPU/Jetson) | 44 MB | 12ms | kernel fusion + half precision |
| TensorRT INT8 (GPU/Jetson) | 22 MB | 7ms | needs calibration data; validate accuracy delta |

The jump from ONNX-on-CPU to TensorRT-FP16 (165ms → 12ms) is the payoff
of pipeline step 4a — and it's a number you can only get by building and
benchmarking the engine, not by reading the PyTorch model's FLOP count.

---

### 3.4 LLM / VLM / Multimodal: Deployment Pipeline → Metrics (15 min)

#### 3.4.1 Typical LLM/VLM Serving Pipeline

Generative models break the "one request in, one response out" model
entirely — a request produces a *sequence* of tokens, generated one at a
time, and the serving stack is built around that:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              TYPICAL LLM/VLM SERVING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. MODEL CHOICE                                                           │
│     Self-hosted open-weight (Llama/Qwen-VL/LLaVA) vs. hosted API           │
│     (Claude/GPT/Gemini) — trades infra ownership for per-token pricing     │
│                          │                                                 │
│                          ▼ (if self-hosting)                               │
│  2. COMPRESS              Quantize weights (AWQ/GPTQ INT4, FP8) to fit     │
│                            more of the model + KV cache in GPU memory      │
│                          │                                                 │
│                          ▼                                                 │
│  3. MULTIMODAL INPUT       For VLM: image → vision tower (e.g. this        │
│     (VLM only)              repo's VisionEncoder) → projector → a          │
│                              sequence of "image tokens" prepended to the   │
│                              text token stream feeding the LLM decoder     │
│                          │                                                 │
│                          ▼                                                 │
│  4. SERVE                 A serving engine built for autoregressive        │
│     vLLM / TensorRT-LLM /  generation, not a generic model server:         │
│     TGI / Triton            • PAGED KV CACHE — manages the growing         │
│                                per-request attention cache without         │
│                                fragmenting GPU memory                      │
│                              • CONTINUOUS BATCHING — new requests join     │
│                                an in-flight batch each decode step,        │
│                                instead of waiting for the batch to finish  │
│                              • (optional) speculative decoding — a small   │
│                                draft model proposes tokens the big model   │
│                                verifies in parallel, cutting latency       │
│                          │                                                 │
│                          ▼                                                 │
│  5. STREAM                Tokens streamed back over SSE/WebSocket as       │
│                            they're generated — the client sees output      │
│                            incrementally, not after the full response      │
│                          │                                                 │
│                          ▼                                                 │
│  6. OBSERVE                Track cost per 1M tokens (input vs. output      │
│                             priced differently), GPU memory dominated by   │
│                             KV cache (not weights) at high concurrency     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why this pipeline looks so different:** a SaaS embedding request is one
fixed-size forward pass; an LLM request is N sequential forward passes
(one per output token), each attending to a KV cache that grows with
every step. That's why LLM serving engines exist as their own category
(vLLM, TensorRT-LLM, TGI) instead of reusing a generic model server.

#### 3.4.2 LLM/VLM Inference Metrics

Traditional latency/throughput still apply, but generation adds metrics
with no equivalent in single-forward-pass inference:

| Metric | What It Captures | Example Target |
|---|---|---|
| **TTFT (Time To First Token)** | Perceived responsiveness — how long until *anything* appears | < 300ms |
| **TPOT (Time Per Output Token)** | Steady-state generation speed, aka inter-token latency | < 30ms/token (~33 tok/s) |
| **Total latency** | `TTFT + TPOT × num_output_tokens` | Depends on response length |
| **Tokens/sec throughput** | Aggregate generation rate across all concurrent requests | GPU- and batch-size-dependent |
| **Cost per 1M tokens** | Input tokens and output tokens are usually priced/costed differently | Track separately, output is pricier |
| **KV cache memory** | Dominant GPU memory cost at scale — grows with context length × batch size | Bounds max concurrency, not weights size |
| **Context window utilization** | How much of the max context prompts actually use | Long prompts shrink available batch size |
| **Vision-encode latency (VLM)** | Time to process image(s) into tokens, separate from text decode | Should not dominate TTFT |
| **Image token count (VLM)** | Patches/tokens an image consumes — competes with text for context budget | e.g. 576 tokens per image tile |
| **Quality metrics** | Task accuracy, hallucination rate, groundedness (VLM) — still required, but tracked separately from system metrics | Task-specific |

#### 3.4.3 Worked Example: Serving a Small VLM with vLLM

```bash
# Self-host a VLM behind vLLM's OpenAI-compatible API, with continuous
# batching and a paged KV cache handled automatically
python -m vllm.entrypoints.openai.api_server \
  --model llava-hf/llava-1.5-7b-hf \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

```python
# Client: measure TTFT and TPOT from a streamed response
import time, requests

start = time.perf_counter()
first_token_time = None
tokens = 0
resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "llava-hf/llava-1.5-7b-hf",
        "messages": [{"role": "user", "content": "Describe this image."}],
        "stream": True,
    },
    stream=True,
)
for chunk in resp.iter_lines():
    if not chunk:
        continue
    if first_token_time is None:
        first_token_time = time.perf_counter()
    tokens += 1

ttft = (first_token_time - start) * 1000
total = time.perf_counter() - start
tpot = ((total - (first_token_time - start)) / max(tokens - 1, 1)) * 1000
print(f"TTFT: {ttft:.0f}ms  TPOT: {tpot:.1f}ms/token  tokens: {tokens}")
```

**Example results** (illustrative, single 7B-class VLM on one GPU):

| Concurrency | TTFT | TPOT | Tokens/sec (aggregate) | Notes |
|---|---|---|---|---|
| 1 | 180ms | 22ms | 45 | vision-encode adds ~60ms to TTFT |
| 8 (continuous batching) | 210ms | 28ms | 290 | KV cache growing, still healthy |
| 32 | 340ms | 55ms | 580 | approaching GPU memory limit — TPOT degrading |

The TPOT degradation at concurrency 32 is the KV-cache-memory ceiling
from 3.4.1 step 4 showing up directly — the same pattern as the SaaS
autoscaling-lag spike in 3.2.3 and the on-device thermal-throttling
pattern in 3.3.2: **you can't interpret the metric without the pipeline
behind it.**

---

### 3.5 Side-by-Side: SaaS vs. On-Device vs. LLM/VLM (5 min)

```
┌───────────────────────┬───────────────────────┬───────────────────────┬───────────────────────┐
│  Axis                 │  SaaS                  │  On-Device             │  LLM / VLM             │
├───────────────────────┼───────────────────────┼───────────────────────┼───────────────────────┤
│  Deployment unit        Container on Cloud Run/  ONNX→TensorRT engine    Serving engine (vLLM/   │
│                          GKE/Vertex AI            or CoreML/TFLite       TensorRT-LLM) + weights  │
│  "More capacity"         Add replicas             Fixed — optimize the   Add GPUs / better        │
│                          (horizontal)              model instead          batching / quantization │
│  Headline latency        p95/p99 request latency   Warm per-inference ms  TTFT + TPOT              │
│  Headline throughput     QPS                       N/A (single user)     Tokens/sec                │
│  Memory bottleneck       Instance RAM/VRAM          Device RAM (100s MB)  KV cache (grows with      │
│                                                                            context × batch)          │
│  Cost unit               $ per 1K inferences        One-time opt cost +   $ per 1M tokens (input     │
│                                                       battery/thermal      vs. output priced         │
│                                                       budget               differently)              │
│  Update mechanism        Blue/green, canary          OTA model push,      New model version         │
│                          rollout                     staged rollout       behind the same API       │
│  Offline behavior        N/A — always connected      Must work with zero  Usually requires           │
│                                                       network dependency  connectivity (self-host    │
│                                                                            changes this)             │
└───────────────────────┴───────────────────────┴───────────────────────┴───────────────────────┘
```

**The unifying takeaway:** all three tracks care about latency and about
"efficiency," and all three show the same pattern — a metric that looks
fine at low load degrades once a real constraint (autoscaling lag,
thermal throttling, KV cache memory) kicks in. Knowing the deployment
pipeline is what tells you *which* constraint to expect, and where.

---

## 4. Course Project: Instrument This Repo

*Deliverable: deploy this repo's retrieval model as a cloud service, and*
*export it toward an on-device path, producing real deployment-pipeline*
*artifacts and metrics for each — connecting Section 3's two pipelines to*
*real numbers on your own machine/project.*

```
PROJECT STEPS
═══════════════════════════════════════════════════════════════

1. SAAS PATH: BUILD + DEPLOY THE SERVICE
   ├── Wrap src/infer.py's retrieval logic in a minimal FastAPI app
   │   with a /health and /embed endpoint (model loaded once at startup)
   ├── Write the Dockerfile from §3.2.3, build, and run it locally
   ├── (Stretch) Deploy to Cloud Run following §3.2.3's commands
   └── Load-test at 1x, 10x, 50x concurrency; record p50/p95/p99 and QPS
       at each level — does latency degrade gracefully or fall off a cliff?

2. ON-DEVICE PATH: EXPORT + BENCHMARK
   ├── Export the vision tower (VisionEncoder + ProjectionHead) to ONNX
   │   following §3.3.3's export_onnx.py pattern
   ├── Validate ONNX output matches PyTorch output (np.allclose)
   ├── Benchmark ONNX Runtime CPU latency (batch=1) as your baseline
   └── (Stretch) If you have access to an NVIDIA GPU, build a TensorRT
       FP16 engine with trtexec and compare latency + size against ONNX

3. WRITE IT UP
   └── One page: your SaaS service's p50/p95/p99 at each concurrency
       level, your ONNX (and TensorRT, if attempted) latency/size numbers,
       and which metric YOU would pick as this project's headline SLO for
       each deployment target, and why
```

---

## 5. Metrics Cheat Sheet

| Metric | Formula / Unit | SaaS Target (example) | On-Device Target (example) | LLM/VLM Target (example) |
|---|---|---|---|---|
| Latency | ms | p99 < 300ms | Warm inference < 50ms | TTFT < 300ms |
| Steady-state speed | — | p50 latency | Warm latency | TPOT < 30ms/token |
| Throughput | reqs/sec or tokens/sec | 2,000 QPS peak | 1 (single user) | Tokens/sec, batch-dependent |
| Concurrency | L = λ · W (Little's Law) | Sized to 3x peak headroom | N/A | Bounded by KV cache memory |
| Cost | $ per unit | $ per 1K inferences | One-time opt cost + battery | $ per 1M tokens (in ≠ out) |
| Model size | MB on disk | Usually not a constraint | < 20 MB (OTA-friendly) | GBs; quantized to fit GPU |
| Peak memory | MB/GB | Bounded by instance type | < 150 MB (low-end device) | Dominated by KV cache |
| Availability | % uptime | 99.9%+ | Must work 100% offline | Depends on self-host vs. API |
| Accuracy delta | vs. FP32/full baseline | Rare to quantize aggressively | < 1-2 pt after INT8 quant | Task accuracy + hallucination rate |

---

## 6. Course Discussion

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DISCUSSION TOPICS FOR CLASS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. PIPELINE BEFORE METRICS                                                 │
│     • Each worked example showed a metric degrading at some threshold      │
│       (autoscaling lag, TensorRT engine limits, KV cache memory). Why      │
│       can't you predict that threshold from the metric alone?             │
│                                                                             │
│  2. YOUR TRACK                                                              │
│     • Which track are you most drawn to — SaaS, Edge/on-device, or         │
│       LLM/VLM serving? What in this class confirmed or changed that?      │
│                                                                             │
│  3. METRICS TRADE-OFFS                                                      │
│     • If you could only report ONE metric for each of the three tracks,    │
│       which would you pick, and what would it hide?                       │
│                                                                             │
│  4. MEASUREMENT HONESTY                                                     │
│     • Why is a TensorRT benchmark on your laptop's GPU misleading for a    │
│       Jetson deployment? Why is an ONNX CPU number misleading for a        │
│       phone's NPU?                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Homework / Next Steps

```
HOMEWORK BEFORE NEXT CLASS (MLE Best Practices in Industry)
═══════════════════════════════════════════════════════════════

1. FINISH THE PROJECT (see Section 4)
   ├── FastAPI service running locally (and, if attempted, on Cloud Run)
   ├── Load-test results at 1x/10x/50x concurrency (p50/p95/p99, QPS)
   ├── ONNX export + validated numerical match against PyTorch
   └── One-page write-up with your chosen headline SLO per track + rationale

2. METRICS PRACTICE
   ├── Practice: run the same load test twice back-to-back — how much
   │   do p95/p99 vary run to run? (This is why you report a
   │   distribution, not a single number.)
   └── Stretch: if you have GPU access, build the TensorRT engine from
       §3.3.3 and report the size/latency comparison table

3. READING FOR CLASS 5
   ├── Skim: Google's "Rules of Machine Learning" (production ML
   │   best-practices doc)
   ├── Browse: the vLLM or TensorRT-LLM docs' serving architecture page
   └── Think about: for YOUR project, what's the single decision
       (model size vs. accuracy, latency vs. cost, etc.) you'd most
       want a senior MLE's opinion on?

RESOURCES:
─────────────────
• Little's Law explainer: https://en.wikipedia.org/wiki/Little%27s_law
• Cloud Run docs: https://cloud.google.com/run/docs
• ONNX export (PyTorch): https://pytorch.org/docs/stable/onnx.html
• TensorRT docs: https://docs.nvidia.com/deeplearning/tensorrt/
• vLLM docs: https://docs.vllm.ai
• Google Rules of ML: https://developers.google.com/machine-learning/guides/rules-of-ml
```

---

## Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      KEY TAKEAWAYS FROM CLASS 4                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ✓ Accuracy alone doesn't tell you if a model is production-ready —        │
│    latency, throughput, cost, and resource footprint all gate shipping     │
│                                                                             │
│  ✓ The deployment pipeline has to be understood BEFORE the metrics —       │
│    every metric degradation in this class's examples traced back to a     │
│    specific pipeline stage (autoscaling lag, engine precision, KV cache)  │
│                                                                             │
│  ✓ SaaS: build → containerize → serve (Cloud Run/GKE/Vertex AI) → scale →  │
│    observe → canary rollout; metrics center on p95/p99, QPS, concurrency, │
│    and cost per 1K requests                                               │
│                                                                             │
│  ✓ On-device: train → simplify → export to ONNX → convert to TensorRT     │
│    (GPU/Jetson) or CoreML/TFLite (mobile) → benchmark on TARGET hardware; │
│    metrics center on model size, memory, latency, and energy per inference│
│                                                                             │
│  ✓ LLM/VLM: compress → serve via a generation-aware engine (vLLM/         │
│    TensorRT-LLM) with continuous batching + paged KV cache → stream       │
│    tokens; metrics center on TTFT, TPOT, tokens/sec, and cost per 1M      │
│    tokens — with a separate vision-encode cost for VLMs                   │
│                                                                             │
│  ✓ Always report latency (and TPOT) as a distribution, never a single     │
│    average — the tail is what breaks SLAs and user trust                  │
│                                                                             │
│  ✓ Never trust a benchmark measured on the wrong hardware — desktop CPU   │
│    numbers don't predict phone NPU performance or Jetson GPU performance  │
│                                                                             │
│  ✓ Next class: MLE best practices — problem understanding, model          │
│    selection, and trade-offs in decision making                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Contact & Questions

**Instructor**: [Your Name]
**Email**: [Your Email]
**Office Hours**: [Schedule]
**Course Repo**: [GitHub Link]

---

*This document was created for educational purposes. Feel free to share and adapt with attribution.*

**Last Updated**: 2026-09-17
**Version**: 2.0
