"""Module 4: Post-training quantization benchmark (FP32 vs INT8)."""

import time
import os
import json

import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import quantize_fx as qfx, QConfigMapping, get_default_qconfig


def create_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def benchmark(model, x, warmup=10, runs=100):
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            model(x)
            times.append((time.perf_counter() - t0) * 1000)
    mean = sum(times) / len(times)
    return round(mean, 2)


def model_size_mb(model):
    path = "/tmp/_size_check.pt"
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / (1024 * 1024)
    os.remove(path)
    return round(size, 2)


def quantize_static(model, cal_data):
    model.eval()
    qconfig = QConfigMapping().set_global(get_default_qconfig("x86"))
    prepared = qfx.prepare_fx(model, qconfig, cal_data[0])
    with torch.no_grad():
        for batch in cal_data:
            prepared(batch)
    return qfx.convert_fx(prepared)


def main():
    print("=" * 50)
    print("MODULE 4: Quantization Benchmark (FX Static PTQ)")
    print("=" * 50)

    model_fp32 = create_resnet18()
    model_fp32.eval()

    # Calibration
    torch.manual_seed(42)
    cal_data = [torch.randn(32, 3, 224, 224) for _ in range(10)]
    model_int8 = quantize_static(model_fp32, cal_data)

    # Size
    size_fp32 = model_size_mb(model_fp32)
    size_int8 = model_size_mb(model_int8)
    reduction = round((1 - size_int8 / size_fp32) * 100, 1)
    print(f"\nSize: {size_fp32} MB → {size_int8} MB ({reduction}% reduction)")

    # Benchmark
    x1 = torch.randn(1, 3, 224, 224)
    x32 = torch.randn(32, 3, 224, 224)

    fp32_bs1 = benchmark(model_fp32, x1)
    int8_bs1 = benchmark(model_int8, x1)
    fp32_bs32 = benchmark(model_fp32, x32, warmup=5, runs=50)
    int8_bs32 = benchmark(model_int8, x32, warmup=5, runs=50)

    sp1 = round(fp32_bs1 / int8_bs1, 2)
    sp32 = round(fp32_bs32 / int8_bs32, 2)
    print(f"bs=1:  FP32 {fp32_bs1} ms, INT8 {int8_bs1} ms, speedup {sp1}x")
    print(f"bs=32: FP32 {fp32_bs32} ms, INT8 {int8_bs32} ms, speedup {sp32}x")

    # Agreement
    test = torch.randn(100, 3, 224, 224)
    with torch.no_grad():
        p32 = model_fp32(test).argmax(1)
        p8 = model_int8(test).argmax(1)
        agree = (p32 == p8).float().mean().item() * 100
    print(f"Prediction agreement: {agree:.1f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    results = {
        "model": "ResNet18",
        "quantization": "static_ptq_fx",
        "size_fp32_mb": size_fp32,
        "size_int8_mb": size_int8,
        "size_reduction_pct": reduction,
        "cpu_batch1": {
            "fp32_ms": fp32_bs1,
            "int8_ms": int8_bs1,
            "speedup": round(fp32_bs1 / int8_bs1, 2),
        },
        "cpu_batch32": {
            "fp32_ms": fp32_bs32,
            "int8_ms": int8_bs32,
            "speedup": round(fp32_bs32 / int8_bs32, 2),
        },
        "prediction_agreement_pct": round(agree, 1),
    }
    with open("results/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved to results/quantization_results.json")


if __name__ == "__main__":
    main()
