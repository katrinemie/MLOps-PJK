"""Module 3: Print summary table from benchmark results."""

import json
import os

results_file = "results/module3_results.json"
if not os.path.exists(results_file):
    print("No results file found.")
    exit(1)

with open(results_file) as f:
    results = json.load(f)

print("\n" + "=" * 65)
print("MODULE 3: Scalable Training Results")
print("=" * 65)
print(f"{'Config':<20} {'GPUs':>5} {'AMP':>5} {'Time(s)':>8} {'Epoch(s)':>9} {'VRAM(MB)':>9} {'Acc':>6}")
print("-" * 65)

for r in results:
    print(f"{r['label']:<20} {r['gpus']:>5} {'Yes' if r['amp'] else 'No':>5} "
          f"{r['total_time_s']:>8.1f} {r['per_epoch_s']:>9.1f} {r['peak_vram_mb']:>9.0f} {r['val_acc']:>5.1f}%")

# Speedup calculations
if len(results) >= 2:
    baseline = next((r for r in results if r['gpus'] == 1 and not r['amp']), None)
    if baseline:
        print(f"\n--- Speedup vs baseline ({baseline['label']}) ---")
        for r in results:
            if r == baseline:
                continue
            speedup = baseline['per_epoch_s'] / r['per_epoch_s']
            vram_diff = r['peak_vram_mb'] - baseline['peak_vram_mb']
            print(f"  {r['label']}: {speedup:.2f}x speedup, {vram_diff:+.0f} MB VRAM")
