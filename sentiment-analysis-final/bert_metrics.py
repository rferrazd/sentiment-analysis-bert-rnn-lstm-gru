
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ===========
# METRICS
# ===========

# Precision: TP / (TP + FP)
# Recall:  TP / (TP + FN)
def calculate_precision_recall_f1(model, loader, device, label_0, label_4):
    """
    Calculate precision and recall for two labels (label_0, label_4) in a classification task.
    """
    model.eval()
    # Initialize counts
    tp_0, fp_0, fn_0 = 0, 0, 0
    tp_4, fp_4, fn_4 = 0, 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            # For label_0
            tp_0 += ((predictions == label_0) & (labels == label_0)).sum().item()
            fp_0 += ((predictions == label_0) & (labels != label_0)).sum().item()
            fn_0 += ((predictions != label_0) & (labels == label_0)).sum().item()
            # For label_4
            tp_4 += ((predictions == label_4) & (labels == label_4)).sum().item()
            fp_4 += ((predictions == label_4) & (labels != label_4)).sum().item()
            fn_4 += ((predictions != label_4) & (labels == label_4)).sum().item()

    precision_0 = tp_0 / (tp_0 + fp_0) if (tp_0 + fp_0) > 0 else 0.0
    recall_0 = tp_0 / (tp_0 + fn_0) if (tp_0 + fn_0) > 0 else 0.0
    precision_4 = tp_4 / (tp_4 + fp_4) if (tp_4 + fp_4) > 0 else 0.0
    recall_4 = tp_4 / (tp_4 + fn_4) if (tp_4 + fn_4) > 0 else 0.0

    return {
        'precision_label_0': precision_0, 
        'recall_label_0': recall_0,
        'f1_label_0' : 2*(precision_0*recall_0) / (precision_0+recall_0) if (precision_0+recall_0) > 0 else 0.0,
        'precision_label_4': precision_4, 
        'f1_label_4' : 2*(precision_4*recall_4) / (precision_4+recall_4) if (precision_4+recall_4) > 0 else 0.0,
        'recall_label_4': recall_4
    }

# Accuracy
def calculate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total

# Confusion Matrix
def get_confusion_matrix(model, loader, device):
    """
    Get predictions and true labels for confusion matrix calculation.
    """
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_predictions)

def plot_confusion_matrix(y_true, y_pred, class_names, title="Confusion Matrix"):
    """
    Plot a confusion matrix heatmap.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    
    return cm

