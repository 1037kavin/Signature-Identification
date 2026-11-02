"""
Wafer Defect Generator CLI
==========================
Generates synthetic semiconductor wafer maps for any failure signature using
a trained Conditional VAE.

Examples:
    # 1. Generate 6 'center' defect wafer maps
    python dl/generate.py --class center --count 6 --output dl/outputs/center_defect.png

    # 2. Generate a full comparison matrix for all 8 failure classes
    python dl/generate.py --all-classes --count 5 --output dl/outputs/comparison_matrix.png

    # 3. Morph smoothly between two defect types (latent space interpolation)
    python dl/generate.py --morph --from-class center --to-class donut --steps 8 --output dl/outputs/morph_center_donut.png

    # 4. Export synthetic wafers to an .npz file for training downstream ML models
    python dl/generate.py --export-dataset --samples-per-class 500 --output dl/data/synthetic_augmented.npz
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dl.dataset import FAILURE_CLASSES, CLASS_TO_IDX
from dl.models.cvae import ConditionalVAE
from dl.utils import (
    plot_wafer_grid,
    plot_all_classes_generation,
    plot_defect_interpolation
)


def load_model(checkpoint_path: str, device: str = "cpu") -> ConditionalVAE:
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at '{checkpoint_path}'.\n"
            f"Please run 'python dl/train_cvae.py' first to train the model."
        )

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    latent_dim = ckpt.get("latent_dim", 64)
    num_classes = ckpt.get("num_classes", len(FAILURE_CLASSES))

    model = ConditionalVAE(
        img_channels=1,
        num_classes=num_classes,
        latent_dim=latent_dim,
        img_size=32
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"[OK] Loaded trained cVAE checkpoint from: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def export_synthetic_dataset(model, samples_per_class: int, output_path: str, device: str = "cpu"):
    """
    Exports a synthetic wafer dataset in .npz format, ready for ML/DL classifier augmentation.
    """
    print(f"[*] Generating {samples_per_class} synthetic samples for each of the {len(FAILURE_CLASSES)} classes...")
    all_images = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for cls_name in FAILURE_CLASSES:
            cls_idx = CLASS_TO_IDX[cls_name]
            wafers = model.sample(class_idx=cls_idx, n_samples=samples_per_class, device=device)
            all_images.append(wafers.cpu().numpy())
            all_labels.extend([cls_idx] * samples_per_class)

    images_arr = np.concatenate(all_images, axis=0)  # Shape: (total, 1, 32, 32)
    labels_arr = np.array(all_labels, dtype=np.int64)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez_compressed(
        output_path,
        images=images_arr,
        labels=labels_arr,
        classes=np.array(FAILURE_CLASSES)
    )
    print(f"[OK] Successfully exported {len(labels_arr):,} synthetic wafer maps to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic wafer maps by failure signature")
    parser.add_argument("--checkpoint", type=str, default="dl/checkpoints/best_cvae.pt", help="Path to cVAE checkpoint")
    parser.add_argument("--class", dest="cls_name", type=str, choices=FAILURE_CLASSES, help="Defect class to generate")
    parser.add_argument("--count", type=int, default=8, help="Number of samples to generate")
    parser.add_argument("--output", type=str, default="dl/outputs/generated_wafers.png", help="Path to save output image")
    parser.add_argument("--all-classes", action="store_true", help="Generate samples for all classes")
    parser.add_argument("--morph", action="store_true", help="Perform latent space morphing between two classes")
    parser.add_argument("--from-class", type=str, default="center", choices=FAILURE_CLASSES)
    parser.add_argument("--to-class", type=str, default="donut", choices=FAILURE_CLASSES)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=1.0, help="Latent sampling temperature (diversity)")
    parser.add_argument("--export-dataset", action="store_true", help="Export synthetic dataset for data augmentation")
    parser.add_argument("--samples-per-class", type=int, default=500, help="Samples per class for export")

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(args.checkpoint, device=device)

    if args.export_dataset:
        export_synthetic_dataset(model, args.samples_per_class, args.output, device=device)
    elif args.morph:
        plot_defect_interpolation(model, args.from_class, args.to_class, steps=args.steps, device=device, save_path=args.output)
    elif args.all_classes:
        plot_all_classes_generation(model, samples_per_class=args.count, device=device, save_path=args.output)
    elif args.cls_name:
        cls_idx = CLASS_TO_IDX[args.cls_name]
        with torch.no_grad():
            wafers = model.sample(cls_idx, n_samples=args.count, device=device, temperature=args.temperature)
        titles = [f"{args.cls_name} #{i+1}" for i in range(args.count)]
        plot_wafer_grid(wafers, titles=titles, nrows=2, ncols=(args.count + 1) // 2, save_path=args.output,
                        suptitle=f"Synthetic Wafer Maps Generated for Failure Class: '{args.cls_name}'")
    else:
        # Default action: all classes
        plot_all_classes_generation(model, samples_per_class=5, device=device, save_path=args.output)


if __name__ == "__main__":
    main()
