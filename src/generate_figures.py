"""Generate report figures from experiment results (reads from JSON)."""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'DAKI4---MLOps-Jonas', 'figures')
RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(OUTDIR, exist_ok=True)

BLUE, RED, GREEN, ORANGE, GRAY = '#2563eb', '#dc2626', '#16a34a', '#ea580c', '#6b7280'

plt.rcParams.update({'font.size': 11, 'figure.dpi': 150, 'savefig.bbox': 'tight'})


def fig_cicd_pipeline():
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.axis('off')

    stages = [
        ('Lint\n(Flake8)', '#eff6ff', BLUE),
        ('Unit Tests\n(pytest)', '#eff6ff', BLUE),
        ('Build Docker\n(commit hash)', '#fef3c7', ORANGE),
        ('Fetch Data\n(DVC/MinIO)', '#fef3c7', ORANGE),
        ('Train\n(MLflow)', '#dcfce7', GREEN),
        ('Evaluate\n(metrics)', '#dcfce7', GREEN),
        ('Register\n(if acc≥80%)', '#fee2e2', RED),
    ]

    w, h = 1.2, 1.4
    y = 0.3
    gap = (10 - len(stages) * w) / (len(stages) + 1)

    for i, (label, bg, edge) in enumerate(stages):
        x = gap + i * (w + gap)
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=edge, linewidth=2))
        ax.text(x + w/2, y + h/2, label, ha='center', va='center', fontsize=8.5, fontweight='bold')
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + w + gap*0.15, y + h/2), xytext=(x + w + 0.02, y + h/2),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    ax.set_title('Jenkins CI/CD Pipeline', fontsize=14, fontweight='bold', pad=15)
    fig.savefig(os.path.join(OUTDIR, 'cicd_pipeline.png'))
    plt.close()
    print('  ✓ cicd_pipeline.png')


def fig_quantization():
    with open(os.path.join(RESULTS, 'quantization_results.json')) as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Size
    ax = axes[0]
    sizes = [data['size_fp32_mb'], data['size_int8_mb']]
    bars = ax.bar(['FP32', 'INT8'], sizes, color=[BLUE, GREEN], width=0.5)
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size')
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val} MB',
                ha='center', fontweight='bold', fontsize=10)
    ax.set_ylim(0, max(sizes) * 1.25)
    ax.text(0.5, 0.92, f'{data["size_reduction_pct"]}% reduction', transform=ax.transAxes,
            ha='center', fontsize=9, color=GREEN, fontweight='bold')

    # Speedup bs=1
    ax = axes[1]
    b1 = data['cpu_batch1']
    times = [b1['fp32_ms'], b1['int8_ms']]
    bars = ax.bar(['FP32', 'INT8'], times, color=[BLUE, GREEN], width=0.5)
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Latency (batch=1)')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.03,
                f'{val} ms', ha='center', fontweight='bold', fontsize=10)
    ax.text(0.5, 0.92, f'{b1["speedup"]}× speedup', transform=ax.transAxes,
            ha='center', fontsize=9, color=GREEN, fontweight='bold')

    # Speedup bs=32
    ax = axes[2]
    b32 = data['cpu_batch32']
    times = [b32['fp32_ms'], b32['int8_ms']]
    bars = ax.bar(['FP32', 'INT8'], times, color=[BLUE, GREEN], width=0.5)
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Latency (batch=32)')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.03,
                f'{val} ms', ha='center', fontweight='bold', fontsize=10)
    ax.text(0.5, 0.92, f'{b32["speedup"]}× speedup', transform=ax.transAxes,
            ha='center', fontsize=9, color=GREEN, fontweight='bold')

    fig.suptitle('D4.1: Post-Training Quantization (FP32 → INT8)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'quantization_comparison.png'))
    plt.close()
    print('  ✓ quantization_comparison.png')


def fig_batch_benchmark():
    with open(os.path.join(RESULTS, 'batch_benchmark_results.json')) as f:
        data = json.load(f)

    fp32 = data['fp32']
    bs = [d['batch_size'] for d in fp32]
    throughput = [d['throughput_fps'] for d in fp32]
    latency = [d['latency_per_image_ms'] for d in fp32]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(bs, throughput, 'o-', color=BLUE, linewidth=2, markersize=7, label='Throughput')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (img/s)', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(bs)
    ax1.set_xticklabels([str(b) for b in bs])

    peak_idx = throughput.index(max(throughput))
    ax1.annotate(f'Peak: {throughput[peak_idx]:.0f} img/s\n(bs={bs[peak_idx]})',
                 xy=(bs[peak_idx], throughput[peak_idx]),
                 xytext=(bs[peak_idx]*1.8, throughput[peak_idx]+15),
                 fontsize=9, fontweight='bold', color=BLUE,
                 arrowprops=dict(arrowstyle='->', color=BLUE))

    ax2 = ax1.twinx()
    ax2.plot(bs, latency, 's--', color=RED, linewidth=2, markersize=6, label='Latency/img')
    ax2.set_ylabel('Latency per image (ms)', color=RED)
    ax2.tick_params(axis='y', labelcolor=RED)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax1.set_title('D4.2: Batch Inference – Throughput vs Latency', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'batch_benchmark.png'))
    plt.close()
    print('  ✓ batch_benchmark.png')


def fig_pruning():
    with open(os.path.join(RESULTS, 'pruning_results.json')) as f:
        data = json.load(f)

    results = data['pruning_results']
    pct = [r['pruning_pct'] for r in results]
    agree = [r['prediction_agreement'] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pct, agree, 'o-', color=BLUE, linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2)

    ax.axvspan(0, 40, alpha=0.06, color=GREEN)
    ax.text(20, 10, 'Safe zone', ha='center', fontsize=9, color=GREEN, style='italic')

    ax.axvspan(45, 55, alpha=0.15, color=RED)
    ax.annotate('Accuracy cliff', xy=(50, 14), fontsize=10, fontweight='bold', color=RED,
                ha='center', xytext=(65, 50), arrowprops=dict(arrowstyle='->', color=RED))

    ax.set_xlabel('Pruning Level (%)')
    ax.set_ylabel('Prediction Agreement (%)')
    ax.set_title('D4.3: L1 Magnitude Pruning on ResNet18', fontweight='bold')
    ax.set_ylim(-5, 110)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'pruning_accuracy.png'))
    plt.close()
    print('  ✓ pruning_accuracy.png')


def fig_finetune():
    with open(os.path.join(RESULTS, 'pruning_results.json')) as f:
        ft = json.load(f)['finetuning']

    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ['70% pruned', '90% pruned']
    before = [ft['70pct']['before'], ft['90pct']['before']]
    after = [ft['70pct']['after'], ft['90pct']['after']]
    x = np.arange(len(labels))
    w = 0.3

    ax.bar(x - w/2, before, w, label='Before', color=RED, alpha=0.8)
    ax.bar(x + w/2, after, w, label='After distillation', color=GREEN, alpha=0.8)

    for i in range(len(labels)):
        ax.text(x[i] - w/2, max(before[i], 2) + 2, f'{before[i]}%',
                ha='center', fontweight='bold', color=RED)
        ax.text(x[i] + w/2, after[i] + 2, f'{after[i]}%',
                ha='center', fontweight='bold', color=GREEN)
        ax.annotate(f'+{after[i]-before[i]:.1f}pp',
                    xy=(x[i], 108), ha='center', fontweight='bold', color=GREEN)

    ax.set_ylabel('Prediction Agreement (%)')
    ax.set_title('D4.4: Recovery via Knowledge Distillation', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 115)
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'finetune_recovery.png'))
    plt.close()
    print('  ✓ finetune_recovery.png')


def fig_gustafson():
    a = 0.15
    gpus = np.arange(1, 9)
    speedup = gpus - a * (gpus - 1)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(gpus, speedup, 'o-', color=BLUE, linewidth=2.5, markersize=8, label='Gustafson')
    ax.plot(gpus, gpus, '--', color=GRAY, linewidth=1, alpha=0.5, label='Linear (ideal)')

    for n in [1, 2, 3]:
        ax.annotate(f'{speedup[n-1]:.2f}×',
                    xy=(n, speedup[n-1]),
                    xytext=(n + 0.3, speedup[n-1] - 0.3),
                    fontsize=9, fontweight='bold', color=BLUE)

    ax.axvline(x=3, color=ORANGE, linestyle=':', alpha=0.5)
    ax.text(3.15, 1, 'AAU cluster\n(3 GPUs)', fontsize=8, color=ORANGE)
    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Speedup')
    ax.set_title("D3.1: Gustafson's Law (a=0.15)", fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'gustafson_speedup.png'))
    plt.close()
    print('  ✓ gustafson_speedup.png')


def fig_scaling_law():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    dims = ['Compute\n(α=0.050)', 'Dataset\n(α=0.095)', 'Parameters\n(α=0.076)']
    factors = [1.1e6, 1500, 9000]
    colors = [RED, ORANGE, BLUE]

    bars = ax.barh(dims, factors, color=colors, height=0.5)
    ax.set_xscale('log')
    ax.set_xlabel('Scale Factor (×)')
    ax.set_title('D3.2: Scaling to Halve Test Loss', fontweight='bold')
    for bar, val in zip(bars, factors):
        label = f'{val:,.0f}×' if val < 1e5 else f'~{val:.1e}×'
        ax.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height()/2, label,
                va='center', fontweight='bold', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'scaling_law.png'))
    plt.close()
    print('  ✓ scaling_law.png')


if __name__ == '__main__':
    print('Generating report figures...\n')
    fig_cicd_pipeline()
    fig_gustafson()
    fig_scaling_law()
    fig_quantization()
    fig_batch_benchmark()
    fig_pruning()
    fig_finetune()
    print(f'\nAll saved to {os.path.abspath(OUTDIR)}/')
