# 🔬 Wafermap Project Analysis & Deep Learning Proposal

## 📁 Project Overview

**Dataset**: WM-811K Wafer Map (`LSWMD.pkl` / `LSWMD.csv`)
- **811,457** total wafer maps with 6 columns
- **25,519** labelled samples (expert-assigned failure labels)
- **785,938** unlabelled samples (`none` class)

**Wafer Map Values**: 0 = no die, 1 = good die, 2 = defective die

---

## 🏷️ Failure Signature Classes (8 Classes)

| Class | Count | Description |
|---|---|---|
| `edge-ring` | 9,680 | Ring of defects at the edge |
| `edge-loc` | 5,189 | Localized edge defects |
| `center` | 4,294 | Central defect cluster |
| `loc` | 3,593 | Random localized spots |
| `scratch` | 1,193 | Linear scratch pattern |
| `random` | 866 | Truly random defects |
| `donut` | 555 | Ring-shaped mid-radius defect |
| `near-full` | 149 | Almost entirely defective |

---

## ✅ What Your Current ML Code Does

### Pipeline
1. **Data Loading & Cleaning** – Unpack nested labels, extract map geometry
2. **Exhaustive EDA** – 12 visualizations (class dist, spatial patterns, defect density)
3. **Feature Engineering (22 Spatial Features)** – Parallelized extraction:
   - Defect density, centroid, std deviation, skewness, kurtosis
   - Edge defect ratio, cluster count, yield rate, map geometry
4. **Class Imbalance Handling** – **SMOTETomek** (oversampling + Tomek link removal)
5. **Dimensionality Reduction** – PCA + t-SNE
6. **Classification (5 baselines → 3 tuned models)**:
   - Logistic Regression, KNN, MLP, Random Forest, Gradient Boosting
   - **Best: Tuned Random Forest** → Accuracy: 0.9592, ROC-AUC: 0.9983
7. **Clustering** – K-Means + DBSCAN

### Direction: `Signature → Failure Class`
> **Input**: 22 hand-crafted spatial features  
> **Output**: One of 8 failure types

---

## 🔄 The "Vice Versa" Task: What You Want Now

### Direction: `Failure Class → Wafer Map` (Conditional Generation)

> **Input**: A failure type label (e.g., "center", "edge-ring")  
> **Output**: A realistic synthetic wafer map image showing that pattern

This is a **Conditional Image Generation** problem — the deep learning model must learn the spatial distribution of defects for each failure class and generate plausible new wafer maps on demand.

---

## 🧠 Deep Learning Architecture Options

### Option 1: Conditional GAN (cGAN) — ⭐ Recommended

**Architecture**: Generator + Discriminator conditioned on class label

```
Label (one-hot) ──┐
                   ├──> Generator ──> Fake Wafer Map
Noise (latent z) ──┘                         │
                                              ▼
                              Discriminator ──> Real/Fake
                                   ▲
                            Real Wafer Map
```

**Why cGAN?**
- Direct conditioning on failure type
- Generates diverse, realistic wafer patterns
- Well-established for image generation
- Works well with small-to-medium datasets

**Variant to try**: **DCGAN** (Deep Convolutional GAN) or **Pix2Pix**

---

### Option 2: Conditional VAE (cVAE)

**Architecture**: Encoder-Decoder with class conditioning

```
Wafer Map + Label ──> Encoder ──> μ, σ ──> Latent z
Label ──────────────────────────────────> Decoder ──> Reconstructed Map
```

**Why cVAE?**
- More stable training than GANs
- Explicit latent space — controllable generation
- Better for small datasets like this one (25K samples)
- Can interpolate between failure modes

---

### Option 3: Conditional Diffusion Model

**Architecture**: Denoising Diffusion Probabilistic Model (DDPM) conditioned on label

**Why Diffusion?**
- State-of-the-art image quality
- Very stable training
- But: slower inference, needs more compute

---

## 🏗️ Recommended Implementation Plan (cVAE)

Given your dataset size (~25K labelled), a **Conditional VAE** is the best starting point — simpler, stable, and interpretable.

### Architecture

```python
# Encoder
Input: Wafer Map (resized to 32×32) + Class Label (one-hot 8-dim)
  → Conv2D(32) → Conv2D(64) → Conv2D(128) → Flatten
  → Dense(256) → [μ, log_σ²]   # Latent dim = 32

# Decoder
[z (32-dim) + Label (8-dim)] → Dense(128×4×4)
  → ConvTranspose2D(128) → ConvTranspose2D(64) → ConvTranspose2D(32)
  → ConvTranspose2D(3)   # Output: 32×32 map
```

### Loss Function
```
Total Loss = Reconstruction Loss (BCE/MSE) + β × KL Divergence
```

### Data Preprocessing
```python
# Normalize wafer maps to [0, 1]
# Resize to fixed size (e.g., 32×32 or 64×64)
# One-hot encode labels
```

---

## 📋 Step-by-Step Implementation Guide

### Step 1: Data Preparation
```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf  # or PyTorch

# Load only labelled data
df_labeled = df[df['failure_label'] != 'none'].copy()

# Resize all wafer maps to fixed shape
def preprocess_wafer(wmap, target_size=(32, 32)):
    wmap = np.array(wmap, dtype=np.float32)
    wmap = tf.image.resize(wmap[..., np.newaxis], target_size).numpy()
    return wmap / 2.0  # Normalize: {0→0, 1→0.5, 2→1.0}

X = np.array([preprocess_wafer(w) for w in df_labeled['waferMap']])
le = LabelEncoder()
y = le.fit_transform(df_labeled['failure_label'])
y_onehot = tf.keras.utils.to_categorical(y, num_classes=8)
```

### Step 2: Build cVAE Model
```python
import tensorflow as tf
from tensorflow.keras import layers

latent_dim = 32
n_classes = 8

# Encoder
def build_encoder():
    map_input = layers.Input(shape=(32, 32, 1))
    label_input = layers.Input(shape=(n_classes,))
    
    x = layers.Conv2D(32, 3, strides=2, activation='relu', padding='same')(map_input)
    x = layers.Conv2D(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Flatten()(x)
    
    label_emb = layers.Dense(64, activation='relu')(label_input)
    x = layers.Concatenate()([x, label_emb])
    x = layers.Dense(256, activation='relu')(x)
    
    z_mean = layers.Dense(latent_dim)(x)
    z_log_var = layers.Dense(latent_dim)(x)
    
    return tf.keras.Model([map_input, label_input], [z_mean, z_log_var])

# Decoder  
def build_decoder():
    z_input = layers.Input(shape=(latent_dim,))
    label_input = layers.Input(shape=(n_classes,))
    
    x = layers.Concatenate()([z_input, label_input])
    x = layers.Dense(8 * 8 * 128, activation='relu')(x)
    x = layers.Reshape((8, 8, 128))(x)
    x = layers.Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(x)
    x = layers.Conv2DTranspose(1, 3, activation='sigmoid', padding='same')(x)
    
    return tf.keras.Model([z_input, label_input], x)
```

### Step 3: Training
```python
encoder = build_encoder()
decoder = build_decoder()
optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        z_mean, z_log_var = encoder([x_batch, y_batch])
        # Reparameterization trick
        eps = tf.random.normal(shape=tf.shape(z_mean))
        z = z_mean + tf.exp(0.5 * z_log_var) * eps
        
        x_reconstructed = decoder([z, y_batch])
        
        # Reconstruction loss
        recon_loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(x_batch, x_reconstructed)
        )
        # KL Divergence
        kl_loss = -0.5 * tf.reduce_mean(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
        )
        total_loss = recon_loss + 0.5 * kl_loss
    
    grads = tape.gradient(total_loss, encoder.trainable_variables + decoder.trainable_variables)
    optimizer.apply_gradients(zip(grads, encoder.trainable_variables + decoder.trainable_variables))
    return total_loss
```

### Step 4: Generation
```python
def generate_wafer_map(failure_class, n_samples=5):
    """Generate synthetic wafer maps for a given failure type."""
    label_idx = le.transform([failure_class])[0]
    label_onehot = tf.keras.utils.to_categorical([label_idx] * n_samples, n_classes)
    
    z_samples = tf.random.normal(shape=(n_samples, latent_dim))
    generated = decoder([z_samples, label_onehot])
    
    return generated.numpy()

# Example: Generate 5 "center" failure maps
center_maps = generate_wafer_map('center', n_samples=5)
```

---

## 🆚 Comparison: Your ML vs. New DL

| Aspect | Your ML (Current) | New DL (Proposed) |
|---|---|---|
| **Task** | Classification | Generation |
| **Input** | 22 features | Failure class label |
| **Output** | Failure class | Wafer map image |
| **Model** | Random Forest, XGBoost | cVAE / cGAN |
| **Training data** | 25K labelled | 25K labelled |
| **Use case** | Quality control, diagnosis | Data augmentation, simulation |
| **Interpretability** | High | Low-Medium |

---

## 🎯 Practical Use Cases for the Generative Model

1. **Data Augmentation** — Generate more `near-full` and `donut` samples (rare classes) to balance your ML training data instead of SMOTE
2. **Process Simulation** — Simulate what defect patterns look like under hypothetical process conditions
3. **Engineer Training** — Generate example wafer maps for training semiconductor engineers
4. **Anomaly Detection** — Reconstruct maps; high reconstruction error = unseen defect type
5. **Transfer Learning** — Pre-train a CNN on synthetic data, then fine-tune on real data

---

## 🛠️ Libraries Required

```bash
pip install tensorflow  # or torch torchvision
pip install numpy pandas matplotlib scikit-learn
```

---

## 💡 Next Steps

1. **Start with cVAE** (stable, fast to implement)
2. Evaluate generation quality visually and with FID score
3. If quality is insufficient, upgrade to **cDCGAN**
4. Use generated data to augment rare classes and re-train your RF/XGBoost models
5. Compare classification performance: SMOTE vs. DL augmentation

> [!TIP]
> For best results, resize wafer maps to **64×64** — larger than 32×32 for more detail, but still manageable computationally.
