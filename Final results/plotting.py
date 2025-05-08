import matplotlib.pyplot as plt
import numpy as np

# Define confusion matrices for two models
matrix_yolo_unet = np.array([
    [160, 5, 5, 10],
    [6, 187, 10, 9],
    [5, 5, 164, 13],
    [2, 0, 0, 0]
])

matrix_mask_rcnn = np.array([
    [79.37, 10.32, 10.32, 0],
    [3.79, 89.39, 6.82, 0],
    [4.24, 20.34, 75.42, 0],
    [0, 0, 0, 0]
])

labels = ["bricks", "concrete", "tiles", "background"]
cmap = "Blues"

# Create subplots for the two confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Function to plot a confusion matrix
def plot_confusion_matrix(ax, matrix, title, is_percentage=False):
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_yticklabels(labels, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=12)

    # Add values inside the matrix
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = f"{matrix[i, j]:.2f}" if is_percentage else int(matrix[i, j])
            color = "white" if i == j else "black"
            ax.text(j, i, value, ha="center", va="center", color=color, fontsize=12)

# Plot YOLO+UNet confusion matrix (raw values)
plot_confusion_matrix(axes[0], matrix_yolo_unet, "YOLO+UNet Confusion Matrix")

# Plot Mask RCNN confusion matrix (percentages)
plot_confusion_matrix(axes[1], matrix_mask_rcnn, "Mask RCNN Confusion Matrix", is_percentage=True)

# Add subplot labels
fig.text(0.25, 0.92, "(a)", fontsize=22, ha="center")
fig.text(0.75, 0.92, "(b)", fontsize=22, ha="center")

# Final layout adjustments
plt.tight_layout(rect=[0, 0, 1, 0.9])
plt.savefig("final_confusion_matrices.png", dpi=300)
plt.show()
