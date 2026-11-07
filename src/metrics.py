import numpy as np
import matplotlib.pyplot as plt


def recall_report(labels, preds, names):
    labels = np.array(labels)
    preds = np.array(preds)
    print(f"{'class':<18}{'recall':>8}{'precision':>11}{'count':>8}")
    for i, name in enumerate(names):
        true_pos = np.sum((labels == i) & (preds == i))
        recall = true_pos / max(np.sum(labels == i), 1)
        precision = true_pos / max(np.sum(preds == i), 1)
        print(f"{name:<18}{recall:>8.3f}{precision:>11.3f}{np.sum(labels == i):>8}")
    print(f"accuracy: {np.mean(labels == preds):.3f}")


def confusion_matrix(labels, preds, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(labels, preds):
        cm[t, p] += 1
    return cm


def plot_confusion(labels, preds, names, path):
    n = len(names)
    cm = confusion_matrix(labels, preds, n)
    norm = cm / cm.sum(axis=1, keepdims=True)  # each row sums to 1

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(names)
    for i in range(n):
        for j in range(n):
            color = "white" if norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", color=color)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()