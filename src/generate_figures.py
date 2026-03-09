"""
Generate report figures from experiment results.
Outputs PNG files for the LaTeX report.
"""

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'DAKI4---MLOps-Jonas', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

RESULTS = os.path.join(os.path.dirname(__file__), '..', 'results')

plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

BLUE = '#2563eb'
RED = '#dc2626'
GREEN = '#16a34a'
ORANGE = '#ea580c'
GRAY = '#6b7280'


def fig_cicd_pipeline():
    """D2.1: CI/CD Pipeline flowchart."""
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
        rect = plt.Rectangle((x, y), w, h, facecolor=bg, edgecolor=edge,
                              linewidth=2, zorder=2, clip_on=False)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=8.5, fontweight='bold', zorder=3)
        if i < len(stages) - 1:
            ax.annotate('', xy=(x + w + gap*0.15, y + h/2),
                        xytext=(x + w + 0.02, y + h/2),
                        arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5))

    ax.text(0.5, y + h + 0.15, 'Git Commit', ha='center', va='bottom',
            fontsize=9, style='italic', color=GRAY)
    ax.annotate('', xy=(gap + 0.1, y + h + 0.1), xytext=(0.5, y + h + 0.15),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1))

    ax.set_title('Jenkins CI/CD Pipeline', fontsize=14, fontweight='bold', pad=15)
    fig.savefig(os.path.join(OUTDIR, 'cicd_pipeline.png'))
    plt.close()
    print('  ✓ cicd_pipeline.png')


def fig_quantization():
    """D4.1: FP32 vs INT8 comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))

    # Size comparison
    ax = axes[0]
    sizes = [42.71, 10.78]
    bars = ax.bar(['FP32', 'INT8'], sizes, color=[BLUE, GREEN], width=0.5, edgecolor='white')
    ax.set_ylabel('Model Size (MB)')
    ax.set_title('Model Size')
    for bar, val in zip(bars, sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val} MB', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.set_ylim(0, 52)
    ax.text(0.5, 0.92, '74.8% reduction', transform=ax.transAxes,
            ha='center', fontsize=9, color=GREEN, fontweight='bold')

    # Speedup bs=1
    ax = axes[1]
    times = [73.53, 6.51]
    bars = ax.bar(['FP32', 'INT8'], times, color=[BLUE, GREEN], width=0.5, edgecolor='white')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Latency (batch=1, CPU)')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{val} ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.text(0.5, 0.92, '11.3× speedup', transform=ax.transAxes,
            ha='center', fontsize=9, color=GREEN, fontweight='bold')

    # Speedup bs=32
    ax = axes[2]
    times = [676.45, 146.22]
    bars = ax.bar(['FP32', 'INT8'], times, color=[BLUE, GREEN], width=0.5, edgecolor='white')
    ax.set_ylabel('Inference Time (ms)')
    ax.set_title('Latency (batch=32, CPU)')
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
                f'{val} ms', ha='center', va='bottom', fontweight='bold', fontsize=10)
    ax.text(0.5, 0.92, '4.6× speedup', transform=ax.transAxes,
            ha='center', fontsize=9, color=GREEN, fontweight='bold')

    fig.suptitle('D4.1: Post-Training Quantization (FP32 → INT8)', fontsize=14, fontweight='bold')
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'quantization_comparison.png'))
    plt.close()
    print('  ✓ quantization_comparison.png')


def fig_batch_benchmark():
    """D4.2: Batch size vs throughput and latency."""
    with open(os.path.join(RESULTS, 'batch_benchmark_results.json')) as f:
        data = json.load(f)

    fp32 = data['fp32']
    bs = [d['batch_size'] for d in fp32]
    throughput = [d['throughput_fps'] for d in fp32]
    latency = [d['latency_per_image_ms'] for d in fp32]

    fig, ax1 = plt.subplots(figsize=(7, 4))

    color1 = BLUE
    ax1.plot(bs, throughput, 'o-', color=color1, linewidth=2, markersize=7, label='Throughput')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Throughput (images/sec)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xscale('log', base=2)
    ax1.set_xticks(bs)
    ax1.set_xticklabels([str(b) for b in bs])

    # Mark peak
    peak_idx = throughput.index(max(throughput))
    ax1.annotate(f'Peak: {throughput[peak_idx]:.0f} img/s\n(bs={bs[peak_idx]})',
                 xy=(bs[peak_idx], throughput[peak_idx]),
                 xytext=(bs[peak_idx]*1.8, throughput[peak_idx]+15),
                 fontsize=9, fontweight='bold', color=color1,
                 arrowprops=dict(arrowstyle='->', color=color1, lw=1.2))

    ax2 = ax1.twinx()
    color2 = RED
    ax2.plot(bs, latency, 's--', color=color2, linewidth=2, markersize=6, label='Latency/image')
    ax2.set_ylabel('Latency per image (ms)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Saturation zone
    ax1.axvspan(16, 64, alpha=0.08, color=RED)
    ax1.text(32, 30, 'Memory\nbandwidth\nbound', ha='center', fontsize=8,
             color=RED, style='italic', alpha=0.8)

    ax1.axvspan(1, 16, alpha=0.06, color=BLUE)
    ax1.text(3, 30, 'Compute\nbound', ha='center', fontsize=8,
             color=BLUE, style='italic', alpha=0.8)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    ax1.set_title('D4.2: Batch Inference – Throughput vs. Latency Tradeoff',
                  fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'batch_benchmark.png'))
    plt.close()
    print('  ✓ batch_benchmark.png')


def fig_pruning():
    """D4.3: Pruning % vs prediction agreement."""
    with open(os.path.join(RESULTS, 'pruning_results.json')) as f:
        data = json.load(f)

    results = data['pruning_results']
    prune_pct = [r['pruning_pct'] for r in results]
    agreement = [r['prediction_agreement'] for r in results]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(prune_pct, agreement, 'o-', color=BLUE, linewidth=2.5, markersize=8,
            markerfacecolor='white', markeredgewidth=2)

    # Highlight cliff
    ax.axvspan(45, 55, alpha=0.15, color=RED)
    ax.annotate('Accuracy\ncliff', xy=(50, 14), fontsize=10,
                fontweight='bold', color=RED, ha='center',
                xytext=(65, 50),
                arrowprops=dict(arrowstyle='->', color=RED, lw=1.5))

    # Safe zone
    ax.axvspan(0, 40, alpha=0.06, color=GREEN)
    ax.text(20, 10, 'Safe zone\n(≤2.5% drop)', ha='center', fontsize=9,
            color=GREEN, style='italic')

    ax.set_xlabel('Pruning Level (%)')
    ax.set_ylabel('Prediction Agreement (%)')
    ax.set_title('D4.3: Effect of L1 Magnitude Pruning on ResNet18',
                 fontweight='bold', fontsize=13)
    ax.set_ylim(-5, 110)
    ax.set_xlim(-2, 100)
    ax.grid(True, alpha=0.3)

    ax.axhline(y=97.5, color=GRAY, linestyle=':', alpha=0.5)
    ax.text(85, 99, '97.5%', fontsize=8, color=GRAY)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'pruning_accuracy.png'))
    plt.close()
    print('  ✓ pruning_accuracy.png')


def fig_finetune():
    """D4.4: Fine-tuning recovery."""
    with open(os.path.join(RESULTS, 'pruning_results.json')) as f:
        data = json.load(f)

    ft = data['finetuning']

    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ['70% pruned', '90% pruned']
    before = [ft['70pct']['before'], ft['90pct']['before']]
    after = [ft['70pct']['after'], ft['90pct']['after']]

    x = np.arange(len(labels))
    w = 0.3

    bars1 = ax.bar(x - w/2, before, w, label='Before fine-tuning', color=RED, alpha=0.8)
    bars2 = ax.bar(x + w/2, after, w, label='After fine-tuning', color=GREEN, alpha=0.8)

    ax.set_ylabel('Prediction Agreement (%)')
    ax.set_title('D4.4: Accuracy Recovery via Knowledge Distillation',
                 fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 115)
    ax.legend(loc='upper left')

    for bar, val in zip(bars1, before):
        ax.text(bar.get_x() + bar.get_width()/2, max(val, 2) + 2,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10, color=RED)
    for bar, val in zip(bars2, after):
        ax.text(bar.get_x() + bar.get_width()/2, val + 2,
                f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=10, color=GREEN)

    # Recovery arrows
    for i in range(len(labels)):
        ax.annotate(f'+{after[i] - before[i]:.1f}pp',
                    xy=(x[i], 108), ha='center', fontsize=10,
                    fontweight='bold', color=GREEN)

    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'finetune_recovery.png'))
    plt.close()
    print('  ✓ finetune_recovery.png')


def fig_gustafson():
    """D3.1: Gustafson speedup estimate."""
    a = 0.15
    gpus = np.arange(1, 9)
    speedup = gpus - a * (gpus - 1)
    efficiency = speedup / gpus * 100

    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax1.plot(gpus, speedup, 'o-', color=BLUE, linewidth=2.5, markersize=8, label='Speedup')
    ax1.plot(gpus, gpus, '--', color=GRAY, linewidth=1, alpha=0.5, label='Linear (ideal)')
    ax1.set_xlabel('Number of GPUs')
    ax1.set_ylabel('Speedup', color=BLUE)
    ax1.tick_params(axis='y', labelcolor=BLUE)

    # Mark cluster GPUs
    for n in [1, 2, 3]:
        ax1.annotate(f'{speedup[n-1]:.2f}×',
                     xy=(n, speedup[n-1]),
                     xytext=(n + 0.3, speedup[n-1] - 0.3),
                     fontsize=9, fontweight='bold', color=BLUE)

    ax1.axvline(x=3, color=ORANGE, linestyle=':', alpha=0.5)
    ax1.text(3.15, 1, 'AAU cluster\n(3 GPUs)', fontsize=8, color=ORANGE)

    ax2 = ax1.twinx()
    ax2.plot(gpus, efficiency, 's--', color=GREEN, linewidth=1.5, markersize=5, alpha=0.7)
    ax2.set_ylabel('Efficiency (%)', color=GREEN)
    ax2.tick_params(axis='y', labelcolor=GREEN)
    ax2.set_ylim(70, 105)

    ax1.set_title("D3.1: Estimated Speedup (Gustafson's Law, a=0.15)",
                  fontweight='bold', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(OUTDIR, 'gustafson_speedup.png'))
    plt.close()
    print('  ✓ gustafson_speedup.png')


def fig_scaling_law():
    """D3.2: Scaling required to halve test loss."""
    fig, ax = plt.subplots(figsize=(7, 3.5))

    dims = ['Compute\n(α=0.050)', 'Dataset Size\n(α=0.095)', 'Parameters\n(α=0.076)']
    factors = [1.1e6, 1500, 9000]
    colors = [RED, ORANGE, BLUE]

    bars = ax.barh(dims, factors, color=colors, height=0.5, edgecolor='white')
    ax.set_xscale('log')
    ax.set_xlabel('Scale Factor Required (×)')
    ax.set_title('D3.2: Scaling Required to Halve Test Loss (Power Law)',
                 fontweight='bold', fontsize=12)

    for bar, val in zip(bars, factors):
        label = f'{val:,.0f}×' if val < 1e5 else f'~{val:.1e}×'
        ax.text(bar.get_width() * 1.3, bar.get_y() + bar.get_height()/2,
                label, va='center', fontweight='bold', fontsize=10)

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
    print(f'\nAll figures saved to {os.path.abspath(OUTDIR)}/')
