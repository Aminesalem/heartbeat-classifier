import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/mitbih_test.csv", delimiter=",", dtype=np.float32)
names = ["normal", "supraventricular", "ventricular", "fusion", "unknown"]

fig, axes = plt.subplots(1, 5, figsize=(15, 2.5))
for i, ax in enumerate(axes):
    beat = data[data[:, -1] == i][0, :-1]
    ax.plot(beat)
    ax.set_title(names[i])
    ax.set_xticks([])
plt.tight_layout()
plt.savefig("results/beats.png", dpi=150)