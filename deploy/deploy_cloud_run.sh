#!/usr/bin/env bash
# Build, push, and deploy the retrieval service (src/service.py) to Cloud
# Run. See Class 4 tutorial, Part 2.2 for the full pipeline walkthrough
# and Part 2.2's worked-example load-test numbers.
#
# Usage: REGION=us-central1 bash deploy/deploy_cloud_run.sh
set -euo pipefail

PROJECT_ID=$(gcloud config get-value project)
REGION=${REGION:-us-central1}
SERVICE=retrieval-api
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

echo "==> [1/4] Pulling the trained checkpoint via DVC (Class 2 tooling)"
dvc pull outputs/fusion_model.pt.dvc

echo "==> [2/4] Building & pushing ${IMAGE} via Cloud Build"
gcloud builds submit --tag "${IMAGE}" -f deploy/Dockerfile .

echo "==> [3/4] Deploying to Cloud Run"
# --min-instances=1 avoids a cold start on the FIRST request of a demo/
#   load test; drop to 0 for a low-traffic tenant to save cost.
# --concurrency=40 is requests handled per instance in parallel -- tune
#   this against your own p95/p99 numbers (see the worked example table),
#   not a guess: too high queues requests behind a slow inference call,
#   too low wastes instances and money.
gcloud run deploy "${SERVICE}" \
  --image "${IMAGE}" \
  --region "${REGION}" \
  --min-instances=1 --max-instances=20 --concurrency=40 \
  --memory=2Gi --cpu=2 \
  --allow-unauthenticated

echo "==> [4/4] Service URL:"
SERVICE_URL=$(gcloud run services describe "${SERVICE}" --region "${REGION}" --format='value(status.url)')
echo "${SERVICE_URL}"

echo ""
echo "Smoke test:"
echo "  curl \"${SERVICE_URL}/health\""
echo "  curl -X POST \"${SERVICE_URL}/embed_text\" -H 'Content-Type: application/json' -d '{\"text\":\"a red circle\"}'"
echo ""
echo "Load test (see Part 2.2 worked example for the concurrency sweep):"
echo "  hey -z 60s -c 50 -m POST -H 'Content-Type: application/json' \\"
echo "      -d '{\"text\":\"a red circle\"}' \"${SERVICE_URL}/embed_text\""
