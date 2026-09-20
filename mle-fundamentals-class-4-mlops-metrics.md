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
3. [Part 2 — Deployment Pipelines & Inference Metrics by Industry (70 min)](#3-part-2--deployment-pipelines--inference-metrics-by-industry-70-min)
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
│  1:00 ─ 1:20   Part 2.4: LLM/VLM/multimodal — architectures, pipeline,    │
│                 KV cache, and metrics                                     │
│  1:20 ─ 1:25   Part 2.5: Side-by-side comparison                          │
│  1:25 ─ 1:30   Project kickoff + Q&A                                       │
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
| **LLM/VLM deployment** | Name the popular LLM/VLM serving engines and VLM architecture patterns, and explain why decode (not encoding) needs a KV cache |
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

## 3. Part 2 — Deployment Pipelines & Inference Metrics by Industry (70 min)

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

*This whole section deploys ONE real model from this repo — the*
*`FusionCLIPModel` dual encoder in `src/model.py`, checkpointed at*
*`outputs/fusion_model.pt` by `src/train.py`. Every command below is*
*runnable against that checkpoint, not pseudocode.*

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
│  1. Package     Trained checkpoint + a serving handler that wraps          │
│                  FusionCLIPModel.encode_text/encode_image in a request/    │
│                  response API  →  src/service.py                          │
│  2. Containerize Dockerfile: pinned deps, model loaded once at startup,    │
│                  checkpoint pulled via DVC, not baked into the image      │
│                  →  deploy/Dockerfile                                      │
│  3. CI          Build image, run unit + smoke tests (does /health         │
│                  respond, does one /embed_text call succeed), push to     │
│                  Artifact Registry, tagged by commit SHA                   │
│                                                                             │
│  DEPLOY TIME                                                               │
│  ────────────                                                              │
│  4. Serve       Deploy the image behind a managed runtime  →              │
│                  deploy/deploy_cloud_run.sh:                              │
│                    • Cloud Run (used here — simplest: HTTP container,     │
│                      autoscale 0→N, optional GPU)                         │
│                    • GKE + a model server (Triton/TorchServe/BentoML) —   │
│                      more control, needed for custom batching             │
│                    • Vertex AI Endpoints — managed model serving,         │
│                      built-in traffic splitting for canaries              │
│  5. Front        API Gateway / Load Balancer: auth (API keys/OAuth),      │
│                  per-tenant rate limiting, request routing                │
│  6. Scale        Autoscaling policy: min/max replicas, target concurrency │
│                  per instance, scale-to-zero for low-traffic tenants      │
│  7. Observe      Cloud Monitoring + Logging: latency/error dashboards,    │
│                  alerting on SLO burn rate — fed by the structured        │
│                  latency_ms logging already in src/service.py            │
│  8. Roll out     Canary or blue/green: new revision gets 5% of traffic,   │
│                  promoted to 100% only if metrics hold                    │
│                                                                             │
│  Client ──▶ LB/Gateway ──▶ [Revision N-1: 95%] ──▶ Model Server ──▶ (logs, │
│                        └──▶ [Revision N (canary): 5%] ──▶ Model Server     │  metrics)
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why the pipeline matters before the metrics:** p99 latency measured on
a single always-warm replica means nothing once real traffic hits
autoscaled, cold-starting, canary-split infrastructure. The metrics in
3.2.4 are only meaningful in the context of *this* pipeline.

#### 3.2.2 From `FusionCLIPModel` to a Deployable Service

`src/infer.py` already does inference — but it loads the model, reads a
manifest, and exits. A service has to load the model *once* and then
answer requests indefinitely. `src/service.py` (new, in the repo) makes
exactly that change, reusing this repo's own model/config classes:

```python
# src/service.py (excerpt — full file in the repo)
from .config import Config
from .model import FusionCLIPModel

state: dict = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
    cfg = Config(**ckpt["config"])                    # same pattern as infer.py
    model = FusionCLIPModel(
        cfg.vision_model_name, cfg.text_model_name, cfg.embed_dim,
        freeze_vision=cfg.freeze_vision, freeze_text=cfg.freeze_text,
    ).to(cfg.device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    state["model"] = model                             # loaded ONCE, not per-request
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/embed_text")
def embed_text(req: EmbedTextRequest):
    start = time.perf_counter()
    embed = state["model"].encode_text(...)             # the actual inference call
    latency_ms = (time.perf_counter() - start) * 1000    # measured on EVERY request
    return {"embedding": embed.squeeze(0).tolist(), "latency_ms": latency_ms}
```

Run it locally against the checkpoint already in `outputs/`:

```bash
pip install -r requirements.txt -r requirements-deploy.txt
uvicorn src.service:app --host 0.0.0.0 --port 8080

curl http://localhost:8080/health
curl -X POST http://localhost:8080/embed_text \
  -H 'Content-Type: application/json' -d '{"text": "a red circle"}'
```

#### 3.2.3 Deploying to GCP: Cloud Run, Step by Step

`deploy/Dockerfile` and `deploy/deploy_cloud_run.sh` (new, in the repo)
turn that local service into a GCP deployment, using the same GCS/DVC
setup verified in Part 1:

```bash
# deploy/deploy_cloud_run.sh (excerpt — full script in the repo)
dvc pull outputs/fusion_model.pt.dvc                          # step 1: get the checkpoint
gcloud builds submit --tag "${IMAGE}" -f deploy/Dockerfile .    # step 2+3: build & push
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" --region "${REGION}" \
  --min-instances=1 --max-instances=20 --concurrency=40 \       # step 4-6: serve + scale
  --memory=2Gi --cpu=2 --allow-unauthenticated
```

A few decisions worth explaining, not just running:

| Decision in the script | Why |
|---|---|
| `dvc pull` before `docker build`, not `COPY outputs/` in the Dockerfile | Keeps the image reusable across model versions — retraining doesn't require a rebuild, only a redeploy pointing at a new checkpoint |
| `--min-instances=1` | Avoids paying a cold-start penalty (model + HF backbone load) on the very first request of a demo or load test; drop to `0` for a genuinely low-traffic tenant to save cost |
| `--concurrency=40` | Requests one Cloud Run instance handles in parallel. PyTorch inference holds the GIL during a forward pass, so this is tuned against measured p95/p99 (3.2.5), not guessed — too high queues requests behind a slow call, too low wastes instances |
| Structured `latency_ms` field in every log line | Cloud Logging can turn a log field into a Cloud Monitoring metric directly — this is how the dashboards in step 7 actually get built, not a separate instrumentation system |
| `gcloud builds submit` → Artifact Registry, tagged by commit SHA | Every deployed revision is traceable back to the exact code + model version that produced it — required for the canary rollback in step 8 |

Run it:

```bash
REGION=us-central1 bash deploy/deploy_cloud_run.sh
```

#### 3.2.4 SaaS Inference Metrics

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

#### 3.2.5 Worked Example: Load-Testing the Deployed Service

With the service live at the URL `deploy/deploy_cloud_run.sh` printed,
load-test `/embed_text` at increasing concurrency (see Section 4's
project steps for the full sweep):

```bash
SERVICE_URL=$(gcloud run services describe retrieval-api --region us-central1 --format='value(status.url)')

hey -z 60s -c 50 -m POST -H 'Content-Type: application/json' \
  -d '{"text":"a red circle"}' "${SERVICE_URL}/embed_text"
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

*Same real model as 3.2 — `FusionCLIPModel` from `outputs/fusion_model.pt`*
*— but exported for a fixed-hardware target instead of a cloud server.*
*The scripts referenced below (`src/export_onnx.py`,*
*`src/benchmark_onnx.py`, `deploy/export_tensorrt.sh`) are real files in*
*the repo, not pseudocode.*

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

`src/export_onnx.py` (new, in the repo) exports `encode_image` /
`encode_text` — the SAME methods `src/service.py` calls in 3.2 — as
standalone ONNX graphs, then validates the export against PyTorch before
you trust it:

```python
# src/export_onnx.py (excerpt — full file in the repo)
class VisionTower(nn.Module):
    """model.encode_image as a single traceable nn.Module graph."""
    def __init__(self, model: FusionCLIPModel):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.encode_image(pixel_values)          # backbone + head + L2-norm, in one graph

def _validate(torch_module, onnx_path, inputs):
    torch_out = torch_module(*inputs.values()).numpy()
    sess = ort.InferenceSession(onnx_path.as_posix())
    onnx_out = sess.run(None, {k: v.numpy() for k, v in inputs.items()})[0]
    assert np.abs(torch_out - onnx_out).max() < 1e-3           # never ship an unvalidated export
```

Run the export, benchmark on CPU, then hand off to TensorRT on GPU/Jetson
hardware:

```bash
# Step 1: export both towers from the trained checkpoint, with validation
python -m src.export_onnx --checkpoint outputs/fusion_model.pt --tower both
#   -> outputs/onnx/vision_tower.onnx, outputs/onnx/text_tower.onnx
#   [vision_tower.onnx] max abs diff PyTorch vs ONNX Runtime: 3.1e-06

# Step 2: CPU baseline latency (runs anywhere, no GPU needed) —
# src/benchmark_onnx.py reports a full p50/p95/p99 distribution, not an average
python -m src.benchmark_onnx --onnx outputs/onnx/vision_tower.onnx \
  --input_name pixel_values --shape 1 3 224 224

# Step 3: on an NVIDIA GPU / Jetson box — build + benchmark a TensorRT engine
bash deploy/export_tensorrt.sh outputs/onnx/vision_tower.onnx outputs/onnx/vision_tower_fp16.engine
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

### 3.4 LLM / VLM / Multimodal: Deployment Pipeline → Metrics (20 min)

#### 3.4.1 Popular LLM/VLM Inference Architectures

Before the pipeline: what you'd actually reach for. LLM/VLM serving has
consolidated around a handful of engines, each built specifically around
the KV-cache/batching problem explained in 3.4.3 — a generic model server
(TorchServe, a plain FastAPI wrapper) does not solve this well:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         POPULAR LLM/VLM INFERENCE (SERVING ENGINE) ARCHITECTURES            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  vLLM                 PagedAttention — KV cache stored in fixed-size       │
│                        "pages" like OS virtual memory, near-zero waste;    │
│                        the default open-source choice for GPU serving      │
│                                                                             │
│  NVIDIA TensorRT-LLM   Kernel-fused, GPU-architecture-specific compiled     │
│                        engines (same idea as TensorRT in 3.3, extended     │
│                        to autoregressive decode + in-flight batching)      │
│                                                                             │
│  Hugging Face TGI      Production server behind HF Inference Endpoints;    │
│                        continuous batching, tensor parallelism             │
│                                                                             │
│  Triton Inference      Multi-framework model server; hosts a               │
│  Server                TensorRT-LLM or vLLM backend for enterprise fleets  │
│                                                                             │
│  SGLang                RadixAttention — shares cached KV-cache PREFIXES    │
│                        across requests; a big win when many requests       │
│                        share a system prompt                              │
│                                                                             │
│  llama.cpp / GGUF      CPU- and edge-oriented, quantized (GGUF) —          │
│                        the on-device analogue of ONNX/TensorRT for LLMs    │
│                        (ties directly to Part 2.3's export pipeline)       │
│                                                                             │
│  MLC-LLM               Compiles LLMs to run on phone/browser/edge GPUs     │
│                                                                             │
│  DeepSpeed-Inference /  Tensor + pipeline parallelism across many GPUs     │
│  DeepSpeed-MII          for models too large for one device               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**VLM architecture patterns** — how the image actually reaches the LLM
decoder differs by family:

| Pattern | Example Models | How Vision Enters the LLM |
|---|---|---|
| Dual-encoder + projector (LLaVA-style) | LLaVA, Qwen-VL | Frozen ViT → small MLP/linear projector → image patch embeddings prepended as extra "tokens" ahead of the text tokens, fed through the SAME decoder |
| Cross-attention fusion | Flamingo, Llama 3.2 Vision | Frozen vision tower; new cross-attention layers interleaved in the LLM read image features directly, without expanding the text token sequence |
| Native multimodal (early fusion) | Gemini-class, GPT-4V/5-class, some Qwen2-VL variants | Trained jointly from the start on interleaved image/text tokens — no bolted-on projector |
| Hosted API (architecture opaque) | Claude, GPT, Gemini via API | You call an endpoint; the serving architecture is the provider's concern, not yours |

This repo's `FusionCLIPModel` (`src/model.py`) is a dual encoder too, but
for **retrieval** — two aligned embeddings compared with a dot product,
never generating text. Bolt a small text-decoder head onto its
`VisionEncoder`'s patch tokens instead of pooling them, and you'd have
the LLaVA-style pattern above; that's the architectural bridge between
what this repo builds and what a VLM adds.

#### 3.4.2 Typical LLM/VLM Serving Pipeline

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

#### 3.4.3 Why the KV Cache Is Critical — Decode Needs It, Encoding Doesn't

This is the single idea that explains both step 2's compression choices
and step 4's engine architecture above, so it's worth deriving, not just
naming.

**The problem, without a cache:** in a transformer decoder, generating
token *t* requires attention over every token 1..*t*, which means
computing Key and Value projections for every one of those tokens. If
you recompute K/V from scratch at every decode step, generating a
200-token response means re-running the K/V projections for token 1
two hundred times, token 2 one hundred ninety-nine times, and so on —
total compute grows roughly with the *cube* of sequence length, almost
all of it redundant: token 1's K/V vector never changes once computed.

**The fix:** cache each layer's K and V tensors for every token the
moment they're computed, and simply *append* to that cache as new tokens
are generated. Each new decode step then only has to compute Q/K/V for
the **one new token**, and attends against the cached K/V for everything
before it:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              PREFILL (encode the prompt) vs. DECODE (generate)              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PREFILL — one parallel pass over the whole known input                    │
│  (prompt tokens +, for a VLM, image tokens from 3.4.1's projector)         │
│                                                                             │
│    tokens: [T1 T2 T3 T4 T5] ──▶ one forward pass, all positions at once    │
│                                  ──▶ WRITES K/V for all 5 tokens to cache  │
│                                                                             │
│    Nothing to READ from yet (cache starts empty) — this step behaves       │
│    exactly like an ENCODER: full self-attention over a fixed input,        │
│    computed once, in parallel. Compute-bound, GPU-efficient.               │
│                                                                             │
│  DECODE — one sequential step per output token                             │
│                                                                             │
│    step 1: compute Q/K/V for T6 only ──▶ attend over CACHED K/V(T1..T5)    │
│                                       ──▶ append T6's K/V to the cache     │
│    step 2: compute Q/K/V for T7 only ──▶ attend over CACHED K/V(T1..T6)    │
│                                       ──▶ append T7's K/V to the cache     │
│    ...repeats once per output token...                                    │
│                                                                             │
│    Each step does O(1) new K/V projection work + reads a growing cache —  │
│    memory-bandwidth-bound, not compute-bound. This is TPOT (3.4.4).       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Why encoding — and this repo's dual-encoder model — never needs one:**
an encoder (BERT-style, or this repo's `VisionEncoder`/`TextEncoder` in
`src/model.py`) computes attention over its *entire* input exactly once,
in parallel, and returns an embedding. There is no second, third, or
hundredth forward pass over that same growing sequence to avoid
recomputation for — the one pass already *is* the full computation, at
full GPU parallelism. A KV cache exists to avoid **repeating** work
across many small sequential passes; a single-pass encoder has nothing
to repeat. This is exactly why 3.2/3.3's deployment paths (a cloud
endpoint or an ONNX/TensorRT engine that returns one embedding per call)
never mention a KV cache, while every engine in 3.4.1 is built around
one: **the deciding factor isn't "LLM vs. non-LLM," it's whether
inference is one parallel pass (prefill/encoding) or many sequential
passes over a growing sequence (decoding).**

**Why the cache dominates memory, not just latency** — per-token cache
size is `2 (K and V) × num_layers × num_kv_heads × head_dim × bytes_per_element`.
For a 7B-class model (32 layers, 32 heads, head_dim 128, FP16):
`2 × 32 × 32 × 128 × 2 bytes ≈ 0.5 MB per token`. At a 4K-token context
and batch size 1 that's already ~2GB — before counting the model's own
weights — which is why 3.4.4's "KV cache memory" metric, not model size,
is what actually bounds how many concurrent requests a GPU can serve.

#### 3.4.4 LLM/VLM Inference Metrics

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

#### 3.4.5 Worked Example: Serving a Small VLM with vLLM

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
from 3.4.2 step 4 (and 3.4.3's cache-size formula) showing up directly —
the same pattern as the SaaS autoscaling-lag spike in 3.2.5 and the
on-device thermal-throttling pattern in 3.3.2: **you can't interpret the metric without the pipeline
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

*Deliverable: deploy this repo's `FusionCLIPModel` retrieval model as a*
*cloud service, and export it toward an on-device path, producing real*
*deployment-pipeline artifacts and metrics for each. The code for both*
*paths already exists in the repo (`src/service.py`, `deploy/`,*
*`src/export_onnx.py`, `src/benchmark_onnx.py`) — this project runs and*
*measures it, then extends it.*

```
PROJECT STEPS
═══════════════════════════════════════════════════════════════

0. TRAIN A CHECKPOINT (if you don't have one yet)
   └── python -m src.train   -->   outputs/fusion_model.pt (see Section 3.2 intro)

1. SAAS PATH: RUN + DEPLOY THE SERVICE  (§3.2, code: src/service.py, deploy/)
   ├── pip install -r requirements.txt -r requirements-deploy.txt
   ├── uvicorn src.service:app --port 8080 ; curl /health and /embed_text
   ├── (Stretch) bash deploy/deploy_cloud_run.sh to ship it to Cloud Run
   └── Load-test at 1x, 10x, 50x concurrency (§3.2.5); record p50/p95/p99
       and QPS at each level — does latency degrade gracefully or fall off
       a cliff? At what concurrency does your --concurrency=40 setting
       start to matter?

2. ON-DEVICE PATH: EXPORT + BENCHMARK  (§3.3, code: src/export_onnx.py, src/benchmark_onnx.py)
   ├── python -m src.export_onnx --tower both   (validates against PyTorch automatically)
   ├── python -m src.benchmark_onnx --onnx outputs/onnx/vision_tower.onnx
   └── (Stretch) If you have access to an NVIDIA GPU/Jetson, run
       bash deploy/export_tensorrt.sh and compare latency + size against
       the CPU/ONNX baseline

3. LLM/VLM CONNECTION (discussion, no code required)  (§3.4)
   └── This repo's dual encoder never needs a KV cache (§3.4.3) because
       it's single-pass. Sketch (in your write-up, not code): if you
       bolted a small text-decoder onto the VisionEncoder's patch tokens
       to build a LLaVA-style captioning VLM, which of §3.4.1's serving
       engines would you reach for, and why would a KV cache suddenly
       matter for that model when it never did for the retrieval model?

4. WRITE IT UP
   └── One page: your SaaS service's p50/p95/p99 at each concurrency
       level, your ONNX (and TensorRT, if attempted) latency/size numbers,
       your answer to step 3, and which metric YOU would pick as this
       project's headline SLO for each deployment target, and why
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
│  ✓ The KV cache exists because decoding makes many sequential passes      │
│    over a growing sequence; a single-pass encoder (this repo's dual       │
│    encoder included) never repeats that work, so it never needs one       │
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

**Last Updated**: 2026-09-20
**Version**: 3.0
