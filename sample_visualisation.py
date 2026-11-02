import os
import numpy as np
import matplotlib.pyplot as plt

# Path to preprocessed labeled wafer maps
data_path = os.path.join(os.path.dirname(__file__), "dl", "data", "wafer_labeled_32x32.npz")

if not os.path.exists(data_path):
    print(f"Error: Data file not found at '{data_path}'.")
    exit(1)

data = np.load(data_path)
images = data["images"]       # Shape: (N, 1, 32, 32)
labels = data["labels"]       # Shape: (N,)
class_names = data["class_names"]

print(f"Loaded {len(images)} wafer maps across {len(class_names)} defect classes.")
print("Classes:", list(class_names))

# Pick a representative sample of 25 wafers across various classes
indices = []
samples_per_class = max(1, 25 // len(class_names))
for cls_idx in range(len(class_names)):
    matching = np.where(labels == cls_idx)[0]
    if len(matching) > 0:
        chosen = np.random.choice(matching, size=min(samples_per_class, len(matching)), replace=False)
        indices.extend(chosen)

# If fewer than 25, pad with random samples
if len(indices) < 25:
    remaining = np.setdiff1d(np.arange(len(images)), indices)
    indices.extend(np.random.choice(remaining, size=25 - len(indices), replace=False))
indices = indices[:25]

fig, axes = plt.subplots(5, 5, figsize=(10, 10))
cmap = plt.colormaps["viridis"].resampled(3)

for idx, ax in zip(indices, axes.flat):
    wmap = images[idx, 0]
    cls_name = class_names[labels[idx]]
    im = ax.imshow(wmap, cmap=cmap, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"{cls_name}", fontsize=10, fontweight="bold")
    ax.axis("off")

plt.suptitle("Sample Wafer Defect Signatures (WM-811K)", fontsize=14, fontweight="bold", y=0.98)
plt.tight_layout()

output_path = os.path.join(os.path.dirname(__file__), "sample_output.png")
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved sample visualization to: {output_path}")
plt.close()