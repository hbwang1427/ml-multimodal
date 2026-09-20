# Container image for the SaaS retrieval service (src/service.py).
# See Class 4 tutorial, Part 2.2.
#
# Build from the REPO ROOT (not this directory) so it can COPY src/:
#   docker build -f deploy/Dockerfile -t retrieval-api .
#
# CPU base image -- swap FROM for an nvidia/cuda base + torch+cu wheel if
# this service needs GPU inference at higher throughput than CPU allows.
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt requirements-deploy.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-deploy.txt

COPY src/ src/

# The checkpoint is pulled at deploy time by deploy_cloud_run.sh (dvc pull
# before the image build) rather than baked into the image here, so the
# image doesn't need to be rebuilt every time the model is retrained --
# only when the SERVICE CODE changes. Mount/COPY outputs/ instead if you
# want a fully self-contained, immutable image per model version.
ENV CHECKPOINT_PATH=outputs/fusion_model.pt
ENV PORT=8080
EXPOSE 8080

# Single worker: PyTorch inference is CPU/GPU-bound and holds the GIL
# during a forward pass, so scale via more Cloud Run instances (see
# --max-instances in deploy_cloud_run.sh), not more workers in one
# container, unless you've added request batching in front of the model.
CMD ["uvicorn", "src.service:app", "--host", "0.0.0.0", "--port", "8080"]
