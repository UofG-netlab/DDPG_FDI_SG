import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

models = ["DQN", "DDPG", "LSTM-DDPG"]

accuracy  = [0.583, 0.663, 0.9993]
precision = [0.892, 0.756, 1.0000]
recall    = [0.574, 0.840, 0.9967]
f1_score  = [0.699, 0.796, 0.9983]

metrics = [accuracy, precision, recall, f1_score]
metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]

# FIX: 4 grayscale colors (one for each metric)
colors = ["black", "dimgray", "gray", "lightgray"]

x = np.arange(len(models))
width = 0.2

plt.figure(figsize=(10, 6))

for i, metric in enumerate(metrics):
    plt.bar(x + (i-1.5)*width, metric, width,
            color=colors[i], label=metric_names[i])

plt.xticks(x, models, fontsize=12)
plt.ylabel("Score", fontsize=12)
plt.ylim(0, 1.2)

plt.legend(title="Metrics", fontsize=10)
plt.title("Model Performance Comparison (Grayscale)", fontsize=14)
plt.grid(axis='y', linestyle='--', color='gray', alpha=0.4)

plt.tight_layout()
plt.savefig("performance_metrics.pdf", bbox_inches='tight')
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# FORCE TIMES NEW ROMAN FONT
# -----------------------------
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 16

# Stability metrics
metrics = ["VSS", "TSV", "DS", "SI"]
values = [0.9815, 1.0, 1.0, 0.9926]

# Close radar loop
values += values[:1]
angles = np.linspace(0, 2 * np.pi, len(values))

# Plot
plt.figure(figsize=(6, 6))
ax = plt.subplot(111, polar=True)

ax.plot(angles, values, linewidth=2, color="black")
ax.fill(angles, values, color="gray", alpha=0.25)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(metrics, fontname='Times New Roman')

ax.set_ylim(0, 1.1)
#plt.title("LSTM–DDPG Stability Metrics", fontname='Times New Roman', fontsize=16)

ax.grid(True, color="gray", linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()