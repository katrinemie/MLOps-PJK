"""
Module 4: Model pruning and fine-tuning.
Documents D4.3 (pruning vs accuracy) and D4.4 (fine-tuning recovery).
Uses pretrained ResNet18 (ImageNet, 1000 classes) and measures prediction
consistency as a proxy for accuracy degradation under pruning.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision import models


def create_resnet18():
    """Use full ImageNet model (1000 classes) for meaningful evaluation."""
    return models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


def get_reference_predictions(model, test_data):
    """Get predictions from the unpruned model as reference."""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        return outputs.argmax(dim=1), outputs


def evaluate_consistency(model, test_data, ref_preds):
    """Measure how many predictions match the unpruned reference model."""
    model.eval()
    with torch.no_grad():
        outputs = model(test_data)
        preds = outputs.argmax(dim=1)
        agreement = (preds == ref_preds).float().mean().item() * 100
        # Also measure output divergence (MSE of logits)
        mse = ((outputs - ref_outputs)**2).mean().item() if 'ref_outputs' in dir() else 0
    return agreement


def get_sparsity(model):
    total = 0
    zeros = 0
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            w = module.weight.data
            total += w.numel()
            zeros += (w == 0).sum().item()
    return zeros / total if total > 0 else 0


def apply_global_pruning(model, amount):
    params = []
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params.append((module, 'weight'))
    if params:
        prune.global_unstructured(params, pruning_method=prune.L1Unstructured, amount=amount)
    return model


def remove_pruning_masks(model):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            try:
                prune.remove(module, 'weight')
            except ValueError:
                pass
    return model


def finetune_with_distillation(pruned_model, teacher_model, epochs=5, lr=0.001):
    """Fine-tune pruned model using knowledge distillation from unpruned teacher."""
    pruned_model.train()
    teacher_model.eval()
    optimizer = torch.optim.Adam(pruned_model.parameters(), lr=lr)
    kl_loss = nn.KLDivLoss(reduction='batchmean')

    torch.manual_seed(99)

    for epoch in range(epochs):
        total_loss = 0
        for _ in range(20):  # 20 batches per epoch
            batch = torch.randn(16, 3, 224, 224)
            with torch.no_grad():
                teacher_out = teacher_model(batch)
                teacher_probs = torch.softmax(teacher_out / 2.0, dim=1)  # temperature=2

            student_out = pruned_model(batch)
            student_log_probs = torch.log_softmax(student_out / 2.0, dim=1)

            loss = kl_loss(student_log_probs, teacher_probs) * (2.0 ** 2)
            optimizer.zero_grad()
            loss.backward()

            # Zero gradients for pruned weights
            for module in pruned_model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    if hasattr(module, 'weight_mask'):
                        if module.weight.grad is not None:
                            module.weight.grad.data *= module.weight_mask

            optimizer.step()
            total_loss += loss.item()

        if epoch % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {total_loss/20:.4f}")

    return pruned_model


def main():
    print("=" * 60)
    print("MODULE 4: Pruning and Fine-tuning")
    print("=" * 60)

    # Fixed test data for consistent evaluation
    torch.manual_seed(42)
    test_data = torch.randn(200, 3, 224, 224)

    # Reference model (unpruned)
    teacher = create_resnet18()
    teacher.eval()

    with torch.no_grad():
        ref_outputs = teacher(test_data)
        ref_preds = ref_outputs.argmax(dim=1)

    print(f"Reference model: {sum(p.numel() for p in teacher.parameters()):,} parameters")

    # Pruning sweep
    pruning_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]

    print(f"\n{'Pruning':>10} | {'Sparsity':>10} | {'Agreement':>10} | {'Drop':>10}")
    print("-" * 50)

    results = []
    for level in pruning_levels:
        model = create_resnet18()
        if level > 0:
            apply_global_pruning(model, amount=level)

        sparsity = get_sparsity(model)
        model.eval()
        with torch.no_grad():
            outputs = model(test_data)
            preds = outputs.argmax(dim=1)
            agreement = (preds == ref_preds).float().mean().item() * 100

        drop = 100 - agreement

        results.append({
            "pruning_pct": level * 100,
            "sparsity": round(sparsity * 100, 1),
            "prediction_agreement": round(agreement, 1),
            "prediction_drop": round(drop, 1),
        })

        print(f"{level*100:>9.0f}% | {sparsity*100:>9.1f}% | {agreement:>9.1f}% | {drop:>+9.1f}pp")

    # Fine-tune heavily pruned models (D4.4)
    print(f"\n{'='*60}")
    print("Fine-tuning pruned models (knowledge distillation)")
    print("=" * 60)

    ft_results = {}
    for prune_level in [0.7, 0.9]:
        print(f"\n--- {int(prune_level*100)}% pruning ---")
        pruned = create_resnet18()
        apply_global_pruning(pruned, amount=prune_level)

        pruned.eval()
        with torch.no_grad():
            preds_before = pruned(test_data).argmax(dim=1)
            agree_before = (preds_before == ref_preds).float().mean().item() * 100
        print(f"Before fine-tuning: {agree_before:.1f}% agreement")

        pruned = finetune_with_distillation(pruned, teacher, epochs=10, lr=0.0005)

        pruned.eval()
        with torch.no_grad():
            preds_after = pruned(test_data).argmax(dim=1)
            agree_after = (preds_after == ref_preds).float().mean().item() * 100
        print(f"After fine-tuning:  {agree_after:.1f}% agreement")
        print(f"Recovery:           {agree_after - agree_before:+.1f}pp")

        ft_results[f"{int(prune_level*100)}pct"] = {
            "before": round(agree_before, 1),
            "after": round(agree_after, 1),
            "recovery_pp": round(agree_after - agree_before, 1),
        }

    # Save results
    os.makedirs("results", exist_ok=True)
    output = {
        "model": "ResNet18 (ImageNet pretrained, 1000 classes)",
        "test_samples": 200,
        "metric": "prediction_agreement_with_unpruned_model",
        "pruning_results": results,
        "finetuning": ft_results,
    }
    with open("results/pruning_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to results/pruning_results.json")


if __name__ == "__main__":
    main()
