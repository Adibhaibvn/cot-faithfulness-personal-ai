import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

core = pd.read_csv("results/processed/fs_core.csv")
fs = pd.read_csv("results/processed/fs_by_corruption.csv")
ci = pd.read_csv("results/processed/fs_ci.csv")

ORDER = ["llama3.2-3b", "llama3.1-8b-q4", "llama3.1-8b-q8", "llama3.1-8b-cloud",
         "gemma2-9b", "llama3.3-70b-cloud", "gemini-flash"]

# Fig 1: headline FS heatmap
piv = core.pivot(index="model", columns="task", values="fs_core").reindex(
    [m for m in ORDER if m in set(core.model)])
plt.figure(figsize=(7, 5))
sns.heatmap(piv, annot=True, fmt=".2f", cmap="viridis", vmin=0, vmax=1)
plt.title("Faithfulness Score (core corruptions)")
plt.tight_layout(); plt.savefig("paper/figures/fig1_fs_heatmap.png", dpi=200); plt.close()

# Fig 2: by corruption type
plt.figure(figsize=(10, 5))
sns.barplot(data=fs, x="corruption", y="fs", hue="model",
            hue_order=[m for m in ORDER if m in set(fs.model)])
plt.ylabel("Answer-change rate"); plt.xticks(rotation=20)
plt.title("Answer change by corruption type (nonsequitur = control)")
plt.tight_layout(); plt.savefig("paper/figures/fig2_by_corruption.png", dpi=200); plt.close()

# Fig 3: RQ2 quantization comparison with 95% CIs
q = ci[ci.model.isin(["llama3.1-8b-q4", "llama3.1-8b-q8", "llama3.1-8b-cloud"])]
fig, ax = plt.subplots(figsize=(8, 5))
tasks = sorted(q.task.unique()); w = 0.25
for i, m in enumerate(["llama3.1-8b-q4", "llama3.1-8b-q8", "llama3.1-8b-cloud"]):
    s = q[q.model == m].set_index("task").reindex(tasks)
    ax.bar([x + i * w for x in range(len(tasks))], s.fs_core, w, label=m,
           yerr=[s.fs_core - s.ci_low, s.ci_high - s.fs_core], capsize=3)
ax.set_xticks([x + w for x in range(len(tasks))]); ax.set_xticklabels(tasks)
ax.set_ylabel("Faithfulness Score (core)"); ax.set_title("Quantization effect (RQ2)")
ax.legend(); plt.tight_layout()
plt.savefig("paper/figures/fig3_quantization.png", dpi=200); plt.close()
print("Saved figures to paper/figures/")

