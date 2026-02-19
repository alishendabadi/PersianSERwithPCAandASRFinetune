import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def print_CV_results(label_to_category, w_accuracy, u_accuracy, loss,
                     precision, recall, f1, accuracy_per_class, cm,
                     writer=None):

    print("*" * 120)
    print("Cross-Validation Finished. Results:")

    scalar_metrics = {
        "w_accuracy": w_accuracy,
        "u_accuracy": u_accuracy,
        "loss":       loss,
        "precision":  precision,
        "recall":     recall,
        "f1":         f1,
    }

    for metric_name, values in scalar_metrics.items():
        arr = np.array(values)
        print(f"  {metric_name:12s}  mean={arr.mean():.4f}  std={arr.std():.4f}  var={arr.var():.6f}")
        if writer is not None:
            writer.add_scalar(f"cv/{metric_name}/mean", arr.mean(), 1)
            writer.add_scalar(f"cv/{metric_name}/std",  arr.std(),  1)

    print("  Per-class accuracy:")
    for category in range(len(accuracy_per_class[0])):
        accs = np.array([fold[category] for fold in accuracy_per_class])
        cat_name = label_to_category[category]
        print(f"    {cat_name:10s}  mean={accs.mean():.4f}  std={accs.std():.4f}")
        if writer is not None:
            writer.add_scalar(f"cv/per_class_accuracy/{cat_name}/mean", accs.mean(), 1)
            writer.add_scalar(f"cv/per_class_accuracy/{cat_name}/std",  accs.std(),  1)

    # Average confusion matrix across all folds
    if writer is not None and len(cm) > 0:
        avg_cm = np.mean(
            [c.astype(float) / c.sum(axis=1, keepdims=True) for c in cm],
            axis=0
        )
        names = list(label_to_category.values())
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Greens',
                    xticklabels=names, yticklabels=names, ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Average Confusion Matrix (All Folds)')
        fig.tight_layout()
        writer.add_figure("cv/avg_confusion_matrix", fig, 1)
        plt.close(fig)

    print("*" * 120)
