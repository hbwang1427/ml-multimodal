#!/usr/bin/env bash
# Convert an ONNX tower (from `python -m src.export_onnx`) into a
# TensorRT engine and benchmark it on THIS machine's GPU. See Class 4
# tutorial, Part 2.3.
#
# Requires an NVIDIA GPU + TensorRT (`trtexec` on PATH) -- run this on a
# Jetson board or a GPU dev VM, NOT a laptop CPU. `trtexec` ships with
# the TensorRT SDK / the `nvcr.io/nvidia/tensorrt` container image.
#
# Usage:
#   bash deploy/export_tensorrt.sh outputs/onnx/vision_tower.onnx outputs/onnx/vision_tower_fp16.engine
set -euo pipefail

ONNX_PATH=${1:-outputs/onnx/vision_tower.onnx}
ENGINE_PATH=${2:-outputs/onnx/vision_tower_fp16.engine}

if ! command -v trtexec &> /dev/null; then
  echo "trtexec not found. Install the TensorRT SDK, or run this inside" >&2
  echo "  docker run --gpus all -it --rm -v \$(pwd):/workspace nvcr.io/nvidia/tensorrt:24.05-py3" >&2
  exit 1
fi

echo "==> [1/2] Building FP16 TensorRT engine from ${ONNX_PATH}"
trtexec --onnx="${ONNX_PATH}" --saveEngine="${ENGINE_PATH}" --fp16

echo "==> [2/2] Benchmarking the engine (batch=1, 100 timed runs after 20 warmup)"
trtexec --loadEngine="${ENGINE_PATH}" --avgRuns=100 --warmUp=20

echo ""
echo "Compare this latency against the CPU baseline from:"
echo "  python -m src.benchmark_onnx --onnx ${ONNX_PATH}"
echo ""
echo "Optional INT8 build -- needs a representative calibration dataset;"
echo "ALWAYS re-validate accuracy after this step, INT8 is not free:"
BASE="${ENGINE_PATH%.engine}"
echo "  trtexec --onnx=${ONNX_PATH} --saveEngine=${BASE}_int8.engine --int8 --calib=<calibration.cache>"
