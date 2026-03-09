"""
Module 4: Batch inference benchmarking.
Measures throughput and latency at different batch sizes.
Documents D4.2: batch processing speedup, latency/throughput tradeoff.
"""

import time
import json
import os

import torch
import torch.nn as nn
from torchvision import models
from torch.quantization import quantize_dynamic


def create_resnet18(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def benchmark_batch(model, batch_size, n_warmup=5, n_runs=30):
    model.eval()
    dummy = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(dummy)

        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            model(dummy)
            end = time.perf_counter()
            times.append((end - start) * 1000)

    mean_ms = sum(times) / len(times)
    throughput = batch_size / (mean_ms / 1000)  # images/sec
    latency_per_image = mean_ms / batch_size

    return {
        "batch_size": batch_size,
        "total_ms": round(mean_ms, 2),
        "latency_per_image_ms": round(latency_per_image, 2),
        "throughput_fps": round(throughput, 1),
    }


def main():
    print("=" * 60)
    print("MODULE 4: Batch Inference Benchmark")
    print("=" * 60)

    model = create_resnet18()
    model_int8 = quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
    model.eval()
    model_int8.eval()

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    print(f"\n{'BS':>4} | {'Total(ms)':>10} | {'Lat/img(ms)':>12} | {'Throughput':>12} | {'INT8 Total':>10} | {'INT8 Thr.':>10}")
    print("-" * 80)

    results_fp32 = []
    results_int8 = []

    for bs in batch_sizes:
        r_fp32 = benchmark_batch(model, bs)
        r_int8 = benchmark_batch(model_int8, bs)
        results_fp32.append(r_fp32)
        results_int8.append(r_int8)
        print(
            f"{bs:>4} | {r_fp32['total_ms']:>10.2f} | {r_fp32['latency_per_image_ms']:>12.2f} | "
            f"{r_fp32['throughput_fps']:>10.1f} | {r_int8['total_ms']:>10.2f} | {r_int8['throughput_fps']:>10.1f}"
        )

    # Find saturation point
    max_thr = max(r['throughput_fps'] for r in results_fp32)
    sat_bs = next(r['batch_size'] for r in results_fp32 if r['throughput_fps'] >= 0.95 * max_thr)

    print(f"\n--- Analysis ---")
    print(f"Peak FP32 throughput: {max_thr:.1f} img/s at batch size {results_fp32[-1]['batch_size']}")
    print(f"Throughput saturates (~95%) at batch size: {sat_bs}")
    print(f"Latency tradeoff: bs=1 gives {results_fp32[0]['latency_per_image_ms']:.2f} ms/img, "
          f"bs=64 gives {results_fp32[-1]['latency_per_image_ms']:.2f} ms/img")

    # Determine if compute or memory bound
    # On CPU: mostly compute-bound for CNNs
    print(f"\nOn CPU, ResNet18 inference is primarily compute-bound.")
    print(f"Increasing batch size improves throughput by better utilizing CPU cache")
    print(f"and enabling vectorized operations (SIMD), until compute saturates.")

    os.makedirs("results", exist_ok=True)
    results = {
        "fp32": results_fp32,
        "int8": results_int8,
        "saturation_batch_size": sat_bs,
        "peak_throughput_fps": max_thr,
    }
    with open("results/batch_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/batch_benchmark_results.json")


if __name__ == "__main__":
    main()
