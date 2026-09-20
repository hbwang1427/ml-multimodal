"""Benchmark ONNX Runtime latency for an exported tower. Runs entirely on
CPU with no GPU/TensorRT required -- use this as a local stand-in for the
target-hardware benchmark `trtexec` would give you on a Jetson/GPU box
(see deploy/export_tensorrt.sh and Class 4 tutorial Part 2.3).

Reports a full latency distribution (p50/p95/p99), never just an average
-- see Part 2.1's metrics vocabulary for why that matters.

Usage:
    python -m src.benchmark_onnx --onnx outputs/onnx/vision_tower.onnx \\
        --input_name pixel_values --shape 1 3 224 224
    python -m src.benchmark_onnx --onnx outputs/onnx/text_tower.onnx \\
        --input_name input_ids --shape 1 32 --dtype int64 \\
        --extra_input attention_mask:1,32:int64
"""
import argparse
import time

import numpy as np
import onnxruntime as ort


def _make_input(shape: list[int], dtype: str) -> np.ndarray:
    if dtype == "int64":
        return np.random.randint(0, 1000, size=shape).astype(np.int64)
    return np.random.randn(*shape).astype(np.float32)


def _parse_extra_input(spec: str) -> tuple[str, np.ndarray]:
    name, shape_str, dtype = spec.split(":")
    shape = [int(s) for s in shape_str.split(",")]
    return name, _make_input(shape, dtype)


def benchmark(onnx_path: str, input_name: str, shape: list[int], dtype: str,
              extra_inputs: list[str], runs: int, warmup: int):
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    inputs = {input_name: _make_input(shape, dtype)}
    for spec in extra_inputs:
        name, arr = _parse_extra_input(spec)
        inputs[name] = arr

    for _ in range(warmup):
        sess.run(None, inputs)

    latencies = []
    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, inputs)
        latencies.append((time.perf_counter() - start) * 1000)

    arr = np.array(latencies)
    print(
        f"n={runs}  p50={np.percentile(arr, 50):.2f}ms  "
        f"p95={np.percentile(arr, 95):.2f}ms  "
        f"p99={np.percentile(arr, 99):.2f}ms  "
        f"mean={arr.mean():.2f}ms"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="outputs/onnx/vision_tower.onnx")
    parser.add_argument("--input_name", default="pixel_values")
    parser.add_argument("--shape", type=int, nargs="+", default=[1, 3, 224, 224])
    parser.add_argument("--dtype", choices=["float32", "int64"], default="float32")
    parser.add_argument(
        "--extra_input", action="append", default=[],
        help='additional input as "name:d0,d1,...:dtype", e.g. attention_mask:1,32:int64',
    )
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    benchmark(
        args.onnx, args.input_name, args.shape, args.dtype,
        args.extra_input, args.runs, args.warmup,
    )


if __name__ == "__main__":
    main()
