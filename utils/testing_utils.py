import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from utils.representation_loading import batch_representation_loader
from tqdm import tqdm
import numpy as np


def test(model, test_loader, names_of_categories, loss_fn, device, representation_config,
         writer=None, fold_idx=0):
    model.eval()
    test_losses = []
    epoch_labels = []
    epoch_preds = []
    accuracy_per_class = {}
    fold_tag = f"fold_{fold_idx + 1}"

    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_names, targets = batch['file_names'], batch['labels'].long().to(device)
            inputs = batch_representation_loader(input_names, representation_config["layer"], representation_config["path"], representation_config["model"], representation_config["data_set"]).to(device)
            output = model(inputs)
            output = output.logits
            loss = loss_fn(output, targets)

            preds = torch.argmax(torch.softmax(output, dim=-1), dim=-1).to('cpu').detach().numpy().tolist()
            epoch_preds = epoch_preds + preds
            labels = targets.to('cpu').detach().numpy().tolist()
            epoch_labels = epoch_labels + labels
            test_losses.append(loss.data.item())
            inputs, targets, output = inputs.to('cpu'), targets.to('cpu'), output.to('cpu')

    unique_classes = set(epoch_labels)
    for class_label in unique_classes:
        indices = [i for i, y in enumerate(epoch_labels) if y == class_label]
        class_accuracy = accuracy_score([epoch_labels[i] for i in indices], [epoch_preds[i] for i in indices])
        accuracy_per_class[class_label] = class_accuracy

    test_loss  = np.mean(test_losses)
    w_acc      = accuracy_score(epoch_labels, epoch_preds)
    u_acc      = balanced_accuracy_score(epoch_labels, epoch_preds)
    precision  = precision_score(epoch_labels, epoch_preds, average='weighted')
    recall     = recall_score(epoch_labels, epoch_preds, average='weighted')
    f1         = f1_score(epoch_labels, epoch_preds, average='weighted')
    cm         = confusion_matrix(epoch_labels, epoch_preds)

    # Build annotated confusion matrix (percent + raw count)
    cm_percent  = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    annotations = [[f"{p:.2f}\n{n}" for p, n in zip(row_p, row_n)]
                   for row_p, row_n in zip(cm_percent, cm)]

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_percent, annot=annotations, fmt='', cmap='Greens',
                xticklabels=names_of_categories,
                yticklabels=names_of_categories, ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix — Fold {fold_idx + 1}')
    fig.tight_layout()

    print(f'Test Loss:{test_loss:.4f} | WA:{w_acc:.4f} | UA:{u_acc:.4f} | Precision:{precision:.4f} | Recall:{recall:.4f} | F1:{f1:.4f}')
    print("Accuracy per class:")
    for class_label, accuracy in accuracy_per_class.items():
        print(f"  Class {class_label}: {accuracy:.4f}")

    if writer is not None:
        # Scalars — all test metrics indexed by fold number
        writer.add_scalar(f"{fold_tag}/test/loss",       test_loss, fold_idx + 1)
        writer.add_scalar(f"{fold_tag}/test/w_accuracy", w_acc,     fold_idx + 1)
        writer.add_scalar(f"{fold_tag}/test/u_accuracy", u_acc,     fold_idx + 1)
        writer.add_scalar(f"{fold_tag}/test/precision",  precision, fold_idx + 1)
        writer.add_scalar(f"{fold_tag}/test/recall",     recall,    fold_idx + 1)
        writer.add_scalar(f"{fold_tag}/test/f1",         f1,        fold_idx + 1)
        for class_label, acc in accuracy_per_class.items():
            writer.add_scalar(f"{fold_tag}/test/accuracy_class_{class_label}", acc, fold_idx + 1)

        # Confusion matrix figure
        writer.add_figure(f"{fold_tag}/confusion_matrix", fig, fold_idx + 1)

    plt.close(fig)

    return {
        "loss": test_loss,
        "w_accuracy": w_acc,
        "u_accuracy": u_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cm": cm,
        "accuracy_per_class": accuracy_per_class
    }


class CrossValidationTestTracker:
    def __init__(self):
        print("Tracker is Initialized with empty memory")
        self.loss = []
        self.accuracy_per_class = []
        self.w_accuracy = []
        self.u_accuracy = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.cm = []

    def append(self, metrics: dict):
        for metric, value in metrics.items():
            if not hasattr(self, metric):
                raise KeyError(f"Unknown metric '{metric}'")
            lst = getattr(self, metric)
            lst.append(value)
