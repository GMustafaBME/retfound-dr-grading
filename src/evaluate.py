"""
Evaluation Metrics for DR Grading Model

Computes AUROC, sensitivity, specificity, Cohen's Kappa,
and generates classification reports.
"""

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, cohen_kappa_score, classification_report,
    confusion_matrix, accuracy_score
)


def compute_metrics(model, dataloader, device, num_classes=5):
    """Compute comprehensive evaluation metrics.
    
    Args:
        model: Trained PyTorch model
        dataloader: Test DataLoader
        device: torch.device
        num_classes: Number of DR grades
    
    Returns:
        metrics: Dictionary containing all evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "kappa": cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
        "auroc": roc_auc_score(all_labels, all_probs, multi_class="ovr"),
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        )
    }
    
    # Per-class sensitivity and specificity
    cm = metrics["confusion_matrix"]
    for i, grade in enumerate(["No DR", "Mild", "Moderate", "Severe", "Proliferative"]):
        tp = cm[i, i]
        fn = cm[i, :].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = cm.sum() - tp - fn - fp
        metrics[f"sensitivity_{grade}"] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics[f"specificity_{grade}"] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    return metrics


if __name__ == "__main__":
    # TODO: Load model checkpoint and run evaluation
    print("Evaluation script ready.")
