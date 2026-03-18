"""Module 4: Pruning sweep + fine-tuning via knowledge distillation."""

import json
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models


def create_resnet18():
    return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


def apply_global_pruning(model, amount):
    params = [(m, 'weight') for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def get_sparsity(model):
    total, zeros = 0, 0
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            total += m.weight.numel()
            zeros += (m.weight.data == 0).sum().item()
    return zeros / total if total > 0 else 0


def agreement(model, test_data, ref_preds):
    model.eval()
    with torch.no_grad():
        preds = model(test_data).argmax(dim=1)
        return (preds == ref_preds).float().mean().item() * 100


def distill(pruned, teacher, epochs=10, lr=0.0005):
    pruned.train()
    teacher.eval()
    opt = torch.optim.Adam(pruned.parameters(), lr=lr)
    kl = nn.KLDivLoss(reduction='batchmean')
    T = 2.0

    torch.manual_seed(99)
    for epoch in range(epochs):
        total_loss = 0
        for _ in range(20):
            x = torch.randn(16, 3, 224, 224)
            with torch.no_grad():
                t_probs = torch.softmax(teacher(x) / T, dim=1)
            s_log = torch.log_softmax(pruned(x) / T, dim=1)

            loss = kl(s_log, t_probs) * (T ** 2)
            opt.zero_grad()
            loss.backward()

            # Keep pruned weights at zero
            for m in pruned.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)) and hasattr(m, 'weight_mask'):
                    if m.weight.grad is not None:
                        m.weight.grad.data *= m.weight_mask

            opt.step()
            total_loss += loss.item()

        if epoch % 3 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, loss: {total_loss/20:.4f}")
    return pruned


def main():
    print("=" * 50)
    print("MODULE 4: Pruning + Fine-tuning")
    print("=" * 50)

    torch.manual_seed(42)
    test_data = torch.randn(200, 3, 224, 224)

    teacher = create_resnet18()
    teacher.eval()
    with torch.no_grad():
        ref_preds = teacher(test_data).argmax(dim=1)

    # Pruning sweep
    levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    print(f"\n{'Pruning':>8} | {'Sparsity':>8} | {'Agreement':>9}")
    print("-" * 35)

    results = []
    for level in levels:
        model = create_resnet18()
        if level > 0:
            apply_global_pruning(model, level)
        sparsity = get_sparsity(model)
        agree = agreement(model, test_data, ref_preds)
        results.append({
            "pruning_pct": level * 100,
            "sparsity": round(sparsity * 100, 1),
            "prediction_agreement": round(agree, 1),
            "prediction_drop": round(100 - agree, 1),
        })
        print(f"{level*100:>7.0f}% | {sparsity*100:>7.1f}% | {agree:>8.1f}%")

    # Fine-tune heavily pruned models
    print(f"\n{'='*50}")
    print("Fine-tuning with knowledge distillation")
    print("=" * 50)

    ft_results = {}
    for level in [0.7, 0.9]:
        print(f"\n--- {int(level*100)}% pruning ---")
        pruned = create_resnet18()
        apply_global_pruning(pruned, level)

        before = agreement(pruned, test_data, ref_preds)
        print(f"Before: {before:.1f}%")

        pruned = distill(pruned, teacher)

        after = agreement(pruned, test_data, ref_preds)
        print(f"After:  {after:.1f}% (recovery: {after - before:+.1f}pp)")

        ft_results[f"{int(level*100)}pct"] = {
            "before": round(before, 1),
            "after": round(after, 1),
            "recovery_pp": round(after - before, 1),
        }

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/pruning_results.json", "w") as f:
        json.dump({
            "model": "ResNet18 (ImageNet, 1000 classes)",
            "test_samples": 200,
            "metric": "prediction_agreement_with_unpruned_model",
            "pruning_results": results,
            "finetuning": ft_results,
        }, f, indent=2)
    print("\nSaved to results/pruning_results.json")


if __name__ == "__main__":
    main()
