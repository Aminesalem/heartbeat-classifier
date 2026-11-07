import numpy as np


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