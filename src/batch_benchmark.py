"""Module 4: Batch inference benchmark (throughput vs latency)."""

import time
import json
import os

import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import quantize_fx as qfx, QConfigMapping, get_default_qconfig


def create_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def quantize_static(model):
    model.eval()
    qconfig = QConfigMapping().set_global(get_default_qconfig("x86"))
    cal_data = [torch.randn(32, 3, 224, 224) for _ in range(5)]
    prepared = qfx.prepare_fx(model, qconfig, cal_data[0])
    with torch.no_grad():
        for batch in cal_data:
            prepared(batch)
    return qfx.convert_fx(prepared)


def benchmark_batch(model, batch_size, warmup=5, runs=30):
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)

    mean_ms = sum(times) / len(times)
    return {
        "batch_size": batch_size,
        "total_ms": round(mean_ms, 2),
        "latency_per_image_ms": round(mean_ms / batch_size, 2),
        "throughput_fps": round(batch_size / (mean_ms / 1000), 1),
    }


def main():
    print("=" * 50)
    print("MODULE 4: Batch Inference Benchmark")
    print("=" * 50)

    torch.manual_seed(42)
    model_fp32 = create_resnet18()
    model_fp32.eval()
    model_int8 = quantize_static(model_fp32)

    batch_sizes = [1, 2, 4, 8, 16, 32, 64]

    header = f"{'BS':>4} | {'FP32(ms)':>9} | {'Lat/img':>8} | "
    header += f"{'FPS':>8} | {'INT8(ms)':>9} | {'INT8 FPS':>9}"
    print("\n" + header)
    print("-" * 65)

    results_fp32, results_int8 = [], []
    for bs in batch_sizes:
        r_fp32 = benchmark_batch(model_fp32, bs)
        r_int8 = benchmark_batch(model_int8, bs)
        results_fp32.append(r_fp32)
        results_int8.append(r_int8)
        row = (f"{bs:>4} | {r_fp32['total_ms']:>9.2f} | "
               f"{r_fp32['latency_per_image_ms']:>8.2f} | "
               f"{r_fp32['throughput_fps']:>8.1f} | "
               f"{r_int8['total_ms']:>9.2f} | "
               f"{r_int8['throughput_fps']:>9.1f}")
        print(row)

    peak = max(results_fp32, key=lambda r: r['throughput_fps'])
    threshold = 0.95 * peak['throughput_fps']
    sat_bs = next(r['batch_size'] for r in results_fp32 if r['throughput_fps'] >= threshold)

    print(f"\nPeak throughput: {peak['throughput_fps']} img/s at bs={peak['batch_size']}")
    print(f"Saturation (~95%) at bs={sat_bs}")

    os.makedirs("results", exist_ok=True)
    with open("results/batch_benchmark_results.json", "w") as f:
        json.dump({
            "fp32": results_fp32,
            "int8": results_int8,
            "saturation_batch_size": sat_bs,
            "peak_throughput_fps": peak['throughput_fps'],
        }, f, indent=2)
    print("Saved to results/batch_benchmark_results.json")


if __name__ == "__main__":
    main()
