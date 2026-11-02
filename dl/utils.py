"""
Wafer Visualization and Generative Evaluation Utilities
======================================================
Provides color-coded wafer rendering matching semiconductor inspection standards,
grid visualization of generated samples, defect morphology interpolation strips,
and checkpoint helpers.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from typing import List, Optional, Union
from dl.dataset import FAILURE_CLASSES, CLASS_TO_IDX, IDX_TO_CLASS

# Semiconductor Inspection Colormap:
# 0.0 -> Background (dark slate)
# 0.5 -> Normal Die (teal/green)
# 1.0 -> Defect Die (bright red)
WAFER_CMAP = plt.cm.plasma


def render_wafer(wmap: Union[np.ndarray, torch.Tensor], ax: Optional[plt.Axes] = None, title: Optional[str] = None):
    """
    Renders a single wafer map image with semiconductor inspection aesthetics.
    """
    if isinstance(wmap, torch.Tensor):
        wmap = wmap.detach().cpu().squeeze().numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(3, 3))
    
    im = ax.imshow(wmap, cmap='magma', vmin=0.0, vmax=1.0, interpolation='nearest')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold', pad=4)
    ax.axis('off')
    return ax


def plot_wafer_grid(
    wafers: Union[np.ndarray, torch.Tensor],
    titles: Optional[List[str]] = None,
    nrows: int = 2,
    ncols: int = 4,
    figsize: tuple = (12, 6),
    save_path: Optional[str] = None,
    suptitle: Optional[str] = None
):
    """
    Plots a multi-panel grid of wafer maps.
    """
    if isinstance(wafers, torch.Tensor):
        wafers = wafers.detach().cpu().squeeze().numpy()
        if wafers.ndim == 2:
            wafers = wafers[np.newaxis, ...]

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).flatten()

    for i in range(len(axes)):
        if i < len(wafers):
            ax = axes[i]
            title = titles[i] if titles and i < len(titles) else None
            render_wafer(wafers[i], ax=ax, title=title)
        else:
            axes[i].axis('off')

    if suptitle:
        plt.suptitle(suptitle, fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Wafer grid saved to: {save_path}")
    else:
        plt.show()


def plot_all_classes_generation(
    model,
    samples_per_class: int = 5,
    device: str = "cpu",
    save_path: Optional[str] = None
):
    """
    Generates a full matrix of synthetic wafer maps across all 8 failure signatures.
    Rows: 8 failure classes
    Columns: Generated samples from distinct latent seeds
    """
    model.eval()
    nrows = len(FAILURE_CLASSES)
    ncols = samples_per_class

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.2, nrows * 2.2))

    for row_idx, cls_name in enumerate(FAILURE_CLASSES):
        class_id = CLASS_TO_IDX[cls_name]
        with torch.no_grad():
            generated = model.sample(class_idx=class_id, n_samples=samples_per_class, device=device)
        
        generated_np = generated.detach().cpu().squeeze().numpy()

        for col_idx in range(samples_per_class):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(generated_np[col_idx], cmap='magma', vmin=0.0, vmax=1.0)
            ax.axis('off')

            if col_idx == 0:
                ax.set_title(cls_name, fontsize=11, fontweight='bold', loc='left')

    plt.suptitle("Synthesized Wafer Maps by Failure Signature (cVAE Generator)", fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Generation comparison matrix saved to: {save_path}")
    else:
        plt.show()


def plot_defect_interpolation(
    model,
    class_a: str,
    class_b: str,
    steps: int = 8,
    device: str = "cpu",
    save_path: Optional[str] = None
):
    """
    Renders continuous latent space interpolation showing the morphing of one defect pattern into another.
    """
    idx_a = CLASS_TO_IDX[class_a]
    idx_b = CLASS_TO_IDX[class_b]

    with torch.no_grad():
        morph = model.interpolate(idx_a, idx_b, steps=steps, device=device)
    
    morph_np = morph.detach().cpu().squeeze().numpy()

    fig, axes = plt.subplots(1, steps, figsize=(steps * 2.0, 2.6))
    for i in range(steps):
        alpha = i / (steps - 1)
        axes[i].imshow(morph_np[i], cmap='magma', vmin=0.0, vmax=1.0)
        axes[i].set_title(f"{alpha:.0%}\n{class_b if alpha > 0.5 else class_a}", fontsize=9)
        axes[i].axis('off')

    plt.suptitle(f"Latent Defect Morphing: {class_a} ➔ {class_b}", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[OK] Interpolation strip saved to: {save_path}")
    else:
        plt.show()
