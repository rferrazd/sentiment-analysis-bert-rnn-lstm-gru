"""
Evaluation Functions for the LSTM , RNN 
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from datetime import datetime
from sklearn.metrics import confusion_matrix, roc_auc_score


def count_trainable_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_time(seconds):
    """
    Format time in seconds to a human-readable string.
    
    Args:
        seconds: Time in seconds (float or int)
        
    Returns:
        str: Formatted time string (e.g., "2h 30m 15s", "45m 30s", "30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def evaluate_model(model, device, loader, label_0, label_4, plot_confusion_matrix=True, title="Confusion Matrix", 
                   save_results=False, output_file=None, model_name="model", dataset_split="test",
                   training_time_seconds=None, num_trainable_parameters=None):
    """
    Single-pass evaluation that computes metrics + (optional) confusion matrix plot.

    Args:
        model: PyTorch model to evaluate
        device: Device to run evaluation on
        loader: DataLoader for the dataset
        label_0: Label index for negative class (0)
        label_4: Label index for positive class (1)
        plot_confusion_matrix: Whether to plot confusion matrix
        title: Title for confusion matrix plot
        save_results: Whether to save results to file
        output_file: Path to save results (if None, auto-generates based on model_name and dataset_split)
        model_name: Name of the model (e.g., "rnn", "lstm", "bert") - used for auto-generating filename
        dataset_split: Name of dataset split (e.g., "train", "val", "test") - used for auto-generating filename
        training_time_seconds: Optional training time in seconds (float). If None, will not be included in results.
        num_trainable_parameters: Optional number of trainable parameters (int). If None, will be calculated from model.

    Notes:
    - The model outputs logits (no softmax). We apply softmax here for probabilities.
    - Dataset labels are expected to be contiguous ids {0,1} (Negative=0, Positive=1).
    """
    model.eval()

    tp_0 = fp_0 = fn_0 = 0
    tp_4 = fp_4 = fn_4 = 0

    all_predictions = []
    all_labels = []
    all_probs = []  # Store probabilities for ROC-AUC calculation

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            logits = model(inputs)  # Shape: (batch_size, num_labels)
            probs = torch.softmax(logits, dim=1) 

            predictions = torch.argmax(logits, dim=1) 

            all_predictions.extend(predictions.detach().cpu().numpy())
            all_labels.extend(labels.detach().cpu().numpy())
            # Store probabilities for positive class (label_4) for ROC-AUC
            all_probs.extend(probs[:, label_4].detach().cpu().numpy())

            tp_0 += ((predictions == label_0) & (labels == label_0)).sum().item()
            fp_0 += ((predictions == label_0) & (labels != label_0)).sum().item()
            fn_0 += ((predictions != label_0) & (labels == label_0)).sum().item()

            tp_4 += ((predictions == label_4) & (labels == label_4)).sum().item()
            fp_4 += ((predictions == label_4) & (labels != label_4)).sum().item()
            fn_4 += ((predictions != label_4) & (labels == label_4)).sum().item()

    total = len(all_labels)
    correct = sum(1 for p, l in zip(all_predictions, all_labels) if p == l)
    accuracy = 100 * correct / total if total > 0 else 0.0

    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
    precision_4 = tp_4 / (tp_4 + fp_4) if (tp_4 + fp_4) > 0 else 0.0

    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
    recall_4 = tp_4 / (tp_4 + fn_4) if (tp_4 + fn_4) > 0 else 0.0

    f1_0 = (
        2 * (precision_0 * recall_0) / (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0
        else 0.0
    )
    f1_4 = (
        2 * (precision_4 * recall_4) / (precision_4 + recall_4)
        if (precision_4 + recall_4) > 0
        else 0.0
    )

    # Calculate ROC-AUC Score
    # For binary classification, we use probabilities of the positive class (label_4)
    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        # Handle case where only one class is present in labels
        roc_auc = 0.0

    cm = confusion_matrix(all_labels, all_predictions, labels=[label_0, label_4])
    
    # Extract confusion matrix values
    tn, fp, fn, tp = cm.ravel()

    if plot_confusion_matrix:
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative (0)", "Positive (1)"],
            yticklabels=["Negative (0)", "Positive (1)"],
            cbar_kws={"label": "Count"},
        )
        plt.title(title, fontsize=14, fontweight="bold", pad=20)
        plt.ylabel("True Label", fontsize=12, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12, fontweight="bold")
        plt.tight_layout()
        plt.show()

        print("\nConfusion Matrix Summary:")
        print(f"True Negatives (TN):  {tn:6d}  - Correctly predicted 0")
        print(f"False Positives (FP): {fp:6d}  - Predicted 1 but actual was 0")
        print(f"False Negatives (FN):  {fn:6d}  - Predicted 0 but actual was 1")
        print(f"True Positives (TP):  {tp:6d}  - Correctly predicted 1")
        print(f"Total samples: {tn + fp + fn + tp}")

    # Calculate or use provided number of trainable parameters
    if num_trainable_parameters is None:
        num_trainable_parameters = count_trainable_parameters(model)
    
    # Prepare results dictionary
    results = {
        "model_name": model_name,
        "dataset_split": dataset_split,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": {
            "accuracy": accuracy,
            "precision_label_0": precision_0,
            "recall_label_0": recall_0,
            "f1_label_0": f1_0,
            "precision_label_4": precision_4,
            "recall_label_4": recall_4,
            "f1_label_4": f1_4,
            "roc_auc_score": roc_auc,
        },
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "total_samples": int(tn + fp + fn + tp)
        },
        "computational_requirements": {
            "num_trainable_parameters": int(num_trainable_parameters),
        }
    }
    
    # Add training time if provided
    if training_time_seconds is not None:
        results["computational_requirements"]["training_time_seconds"] = float(training_time_seconds)
        results["computational_requirements"]["training_time_formatted"] = format_time(training_time_seconds)

    # Save results to file if requested
    if save_results:
        if output_file is None:
            # Auto-generate filename: results/{model_name}_{dataset_split}_metrics.json
            os.makedirs("results", exist_ok=True)
            output_file = f"results/{model_name}_{dataset_split}_metrics.json"
        
        # Convert numpy types to native Python types for JSON serialization
        results_json = {
            "model_name": results["model_name"],
            "dataset_split": results["dataset_split"],
            "timestamp": results["timestamp"],
            "metrics": {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                       for k, v in results["metrics"].items()},
            "confusion_matrix": {k: int(v) for k, v in results["confusion_matrix"].items()},
            "computational_requirements": {
                "num_trainable_parameters": int(results["computational_requirements"]["num_trainable_parameters"]),
            }
        }
        
        # Add training time if present
        if "training_time_seconds" in results["computational_requirements"]:
            results_json["computational_requirements"]["training_time_seconds"] = float(
                results["computational_requirements"]["training_time_seconds"]
            )
            results_json["computational_requirements"]["training_time_formatted"] = results["computational_requirements"]["training_time_formatted"]
        
        with open(output_file, "w") as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")

    # Print computational requirements if available
    print("\n" + "=" * 60)
    print("COMPUTATIONAL REQUIREMENTS")
    print("=" * 60)
    print(f"Number of trainable parameters: {num_trainable_parameters:,}")
    if training_time_seconds is not None:
        print(f"Training time: {format_time(training_time_seconds)} ({training_time_seconds:.2f} seconds)")
    print("=" * 60)
    
    # Return results (keep confusion_matrix as numpy array for compatibility)
    return {
        "accuracy": accuracy,
        "precision_label_0": precision_0,
        "recall_label_0": recall_0,
        "f1_label_0": f1_0,
        "precision_label_4": precision_4,
        "recall_label_4": recall_4,
        "f1_label_4": f1_4,
        "roc_auc_score": roc_auc,
        "confusion_matrix": cm,
        "num_trainable_parameters": num_trainable_parameters,
        "training_time_seconds": training_time_seconds,
    }
