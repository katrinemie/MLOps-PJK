"""
Module 4: Post-training quantization and inference benchmarking.
Uses FX graph mode quantization (handles quant/dequant stubs automatically).
Compares FP32 vs INT8 on ResNet18.
"""

import time
import os
import json

import torch
import torch.nn as nn
from torchvision import models
from torch.ao.quantization import quantize_fx, QConfigMapping, get_default_qconfig


def create_resnet18(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def benchmark_inference(model, input_tensor, n_warmup=10, n_runs=100, device="cpu"):
    model.eval()
    inp = input_tensor.to(device)
    with torch.no_grad():
        for _ in range(n_warmup):
            model(inp)
        if device == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(inp)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append((end - start) * 1000)

    mean = sum(times) / len(times)
    return {
        "mean_ms": mean,
        "min_ms": min(times),
        "max_ms": max(times),
        "std_ms": (sum((t - mean)**2 for t in times) / len(times)) ** 0.5,
    }


def get_model_size_mb(model):
    tmp = "/tmp/model_size_check.pt"
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / (1024 * 1024)
    os.remove(tmp)
    return size


def quantize_static_fx(model, calibration_data):
    """FX graph mode static quantization - handles ResNet automatically."""
    model.eval()

    qconfig_mapping = QConfigMapping().set_global(
        get_default_qconfig("x86")
    )
    example_input = calibration_data[0]

    # Prepare
    prepared = quantize_fx.prepare_fx(model, qconfig_mapping, example_input)

    # Calibrate
    with torch.no_grad():
        for batch in calibration_data:
            prepared(batch)

    # Convert
    quantized = quantize_fx.convert_fx(prepared)
    return quantized


def main():
    print("=" * 60)
    print("MODULE 4: Post-Training Quantization Benchmark")
    print("(FX Graph Mode Static Quantization)")
    print("=" * 60)

    model_fp32 = create_resnet18(num_classes=2)
    model_fp32.eval()

    # Calibration data
    torch.manual_seed(42)
    calibration_data = [torch.randn(32, 3, 224, 224) for _ in range(10)]

    print("\nApplying FX static quantization with calibration...")
    model_int8 = quantize_static_fx(model_fp32, calibration_data)

    # Sizes
    size_fp32 = get_model_size_mb(model_fp32)
    size_int8 = get_model_size_mb(model_int8)
    print(f"\nModel size FP32: {size_fp32:.2f} MB")
    print(f"Model size INT8: {size_int8:.2f} MB")
    print(f"Size reduction:  {(1 - size_int8/size_fp32) * 100:.1f}%")

    # Benchmark batch=1
    dummy1 = torch.randn(1, 3, 224, 224)
    print("\n--- Single image (batch=1, CPU) ---")
    fp32_t = benchmark_inference(model_fp32, dummy1, device="cpu")
    int8_t = benchmark_inference(model_int8, dummy1, device="cpu")
    sp1 = fp32_t['mean_ms'] / int8_t['mean_ms']
    print(f"FP32: {fp32_t['mean_ms']:.2f} ms (±{fp32_t['std_ms']:.2f})")
    print(f"INT8: {int8_t['mean_ms']:.2f} ms (±{int8_t['std_ms']:.2f})")
    print(f"Speedup: {sp1:.2f}x")

    # Benchmark batch=32
    dummy32 = torch.randn(32, 3, 224, 224)
    print("\n--- Batch (batch=32, CPU) ---")
    fp32_b = benchmark_inference(model_fp32, dummy32, n_warmup=5, n_runs=50, device="cpu")
    int8_b = benchmark_inference(model_int8, dummy32, n_warmup=5, n_runs=50, device="cpu")
    sp32 = fp32_b['mean_ms'] / int8_b['mean_ms']
    print(f"FP32: {fp32_b['mean_ms']:.2f} ms (±{fp32_b['std_ms']:.2f})")
    print(f"INT8: {int8_b['mean_ms']:.2f} ms (±{int8_b['std_ms']:.2f})")
    print(f"Speedup: {sp32:.2f}x")

    # GPU FP32 for comparison
    gpu_ms = None
    if torch.cuda.is_available():
        print("\n--- GPU FP32 comparison ---")
        m_gpu = model_fp32.cuda()
        gt = benchmark_inference(m_gpu, dummy1, device="cuda")
        gpu_ms = round(gt['mean_ms'], 2)
        print(f"GPU FP32 (bs=1): {gpu_ms} ms")

    # Accuracy
    print("\n--- Prediction agreement ---")
    test = torch.randn(100, 3, 224, 224)
    with torch.no_grad():
        p_fp32 = model_fp32(test).argmax(1)
        p_int8 = model_int8(test).argmax(1)
        agree = (p_fp32 == p_int8).float().mean().item() * 100
    print(f"Agreement: {agree:.1f}%")

    # Save
    os.makedirs("results", exist_ok=True)
    results = {
        "model": "ResNet18",
        "quantization": "static_ptq_fx",
        "size_fp32_mb": round(size_fp32, 2),
        "size_int8_mb": round(size_int8, 2),
        "size_reduction_pct": round((1 - size_int8/size_fp32) * 100, 1),
        "cpu_batch1": {
            "fp32_ms": round(fp32_t['mean_ms'], 2),
            "int8_ms": round(int8_t['mean_ms'], 2),
            "speedup": round(sp1, 2),
        },
        "cpu_batch32": {
            "fp32_ms": round(fp32_b['mean_ms'], 2),
            "int8_ms": round(int8_b['mean_ms'], 2),
            "speedup": round(sp32, 2),
        },
        "gpu_batch1_fp32_ms": gpu_ms,
        "prediction_agreement_pct": agree,
    }
    with open("results/quantization_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)
    print(f"Size: {size_fp32:.1f} MB → {size_int8:.1f} MB ({results['size_reduction_pct']}% reduction)")
    print(f"Speedup (bs=1):  {sp1:.2f}x")
    print(f"Speedup (bs=32): {sp32:.2f}x")
    print(f"Agreement:       {agree:.1f}%")


if __name__ == "__main__":
    main()
