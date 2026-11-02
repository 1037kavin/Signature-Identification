# 🔬 Wafer Map Deep Learning (Inverse Generative Synthesis)

Welcome to the **Deep Learning (DL)** module for the **WM-811K semiconductor wafer map project**.

---

## 🔄 The Vice-Versa Paradigm

In your previous machine learning pipeline ([`wafermap_fixed.ipynb`](../wafermap_fixed.ipynb)):
$$\text{Wafer Map } X \xrightarrow{\text{22 Spatial Features}} \text{ML Classifier} \xrightarrow{} \text{Failure Class } y$$

In this Deep Learning module, we achieve the **vice-versa (inverse) capability**:
$$\text{Failure Class } y + \text{Latent Noise } z \sim \mathcal{N}(0, I) \xrightarrow{\text{Conditional Generative Model (cVAE / cGAN)}} \text{Realistic Synthetic Wafer Map } \hat{X}$$

Given any defect signature (e.g., `center`, `donut`, `edge-ring`, `scratch`), the model synthesizes new, high-fidelity wafer defect maps on demand.

---

## 📁 Directory Structure

```
wafer/
├── LSWMD.csv                          # Raw dataset summary
├── wafermap_fixed.ipynb               # Original ML Classification Pipeline
└── dl/                                # ⭐ NEW Deep Learning Module
    ├── data/                          # Cached preprocessed arrays (.npz)
    ├── checkpoints/                   # Saved model weights (best_cvae.pt)
    ├── outputs/                       # Generated wafer plots & morphing strips
    ├── models/
    │   ├── __init__.py
    │   ├── cvae.py                    # Conditional Variational Autoencoder
    │   └── cgan.py                    # Conditional GAN (cDCGAN)
    ├── dataset.py                     # Dataset loader, resizing, caching & PyTorch DataLoader
    ├── utils.py                       # Inspection colormaps, wafer grids & morphing plots
    ├── train_cvae.py                  # CLI training script with checkpointing
    ├── generate.py                    # CLI generation tool for on-demand wafer maps
    ├── wafer_generative_dl.ipynb      # Complete interactive Jupyter Notebook
    └── README.md                      # Documentation
```

---

## 🏷️ The 8 Failure Signatures

| Class | Ground Truth Count | Real-World Failure Cause |
|---|---|---|
| `edge-ring` | 9,680 | Bevel/clamp ring defect, CMP slurry buildup |
| `edge-loc` | 5,189 | Wafer edge chipping, chuck pin contact damage |
| `center` | 4,294 | Spin-coating non-uniformity, gas flow stagnation |
| `loc` | 3,593 | Local particle contamination, reticle defect |
| `scratch` | 1,193 | Robotic handling scratch, tweezer mechanical mark |
| `random` | 866 | General airborne cleanroom particulate defect |
| `donut` | 555 | Mid-radius thermal gradient / etching non-uniformity |
| `near-full` | 149 | Severe gross functional failure, catastrophic abort |

---

## 🚀 Quickstart: How to Use

### 1. Interactive Jupyter Notebook
Open and run:
```bash
jupyter notebook dl/wafer_generative_dl.ipynb
```
Follow the step-by-step cells for visual EDA, model training, generation, and defect interpolation.

---

### 2. Train the Model via Command Line
Train a Conditional VAE with automatic checkpointing and visual sample outputs:
```bash
python dl/train_cvae.py --epochs 25 --batch-size 128 --beta 0.5
```

---

### 3. Generate Wafer Maps on Demand

#### Generate 8 samples for a specific failure type:
```bash
python dl/generate.py --class center --count 8 --output dl/outputs/center_defect.png
```

#### Generate a full comparison grid across all 8 classes:
```bash
python dl/generate.py --all-classes --count 5 --output dl/outputs/all_classes.png
```

#### Morph smoothly between two defect types (latent space interpolation):
```bash
python dl/generate.py --morph --from-class center --to-class donut --steps 8 --output dl/outputs/morph_center_donut.png
```

#### Export synthetic wafers to augment downstream ML classifiers:
```bash
python dl/generate.py --export-dataset --samples-per-class 500 --output dl/data/synthetic_augmented.npz
```

---

## 🧠 Model Architecture (cVAE)

```
[Wafer Map (32×32)] ──┐
                      ├──> Conv2D Layers ──> Flatten ──> [μ, log σ²]
[Class Label (OneHot)] ─┘                                     │
                                                  Reparameterization: z = μ + σ ⊙ ε
                                                              │
[Class Label (OneHot)] ───────────────────────┐               │
                                              ├──> Dense ──> ConvTranspose2D ──> [Synthetic Wafer Map (32×32)]
                              [Latent Vector z] ─┘
```

**Loss Function**:
$$\mathcal{L} = \text{BCE}(X, \hat{X}) + \beta \cdot D_{KL}(q(z \mid X, y) \,\|\, \mathcal{N}(0, I))$$
- **Reconstruction Loss (BCE)**: Preserves defect patterns, edges, and die coordinates.
- **KL Divergence Loss**: Shapes the latent space into a smooth continuous Gaussian manifold.
