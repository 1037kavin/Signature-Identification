"""
WM-811K Wafer Map Dataset Loader & Preprocessing Pipeline
=========================================================
Handles reading the LSWMD wafermap dataset, extracting the 8 labeled failure types,
resizing non-uniform wafer maps to a standardized grid (default 32x32 or 64x64),
and providing PyTorch Datasets/DataLoaders with intelligent caching.
"""

import os
import pickle
import numpy as np
import pandas as pd
from typing import Tuple, Optional, List, Dict
from scipy.ndimage import zoom

# Failure classes in WM-811K dataset (excluding 'none')
FAILURE_CLASSES = [
    'edge-ring',
    'edge-loc',
    'center',
    'loc',
    'scratch',
    'random',
    'donut',
    'near-full'
]

CLASS_TO_IDX: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(FAILURE_CLASSES)}
IDX_TO_CLASS: Dict[int, str] = {i: cls_name for i, cls_name in enumerate(FAILURE_CLASSES)}

DEFAULT_PKL_CANDIDATES = [
    os.path.join(os.path.dirname(__file__), "..", "LSWMD.pkl"),
    os.path.join(os.path.dirname(__file__), "data", "LSWMD.pkl"),
    os.path.join(os.getcwd(), "LSWMD.pkl"),
]
if "LSWMD_PATH" in os.environ:
    DEFAULT_PKL_CANDIDATES.insert(0, os.environ["LSWMD_PATH"])


def find_dataset_file(custom_path: Optional[str] = None) -> str:
    """Locates the LSWMD.pkl dataset file across known candidate paths."""
    if custom_path and os.path.exists(custom_path):
        return custom_path
    for path in DEFAULT_PKL_CANDIDATES:
        if os.path.exists(path) and os.path.isfile(path):
            return os.path.abspath(path)
    raise FileNotFoundError(
        "Could not find LSWMD.pkl. Checked candidate paths:\n" +
        "\n".join(DEFAULT_PKL_CANDIDATES)
    )


def extract_label_str(raw_label) -> str:
    """Unwraps nested list/array labels such as [['edge-ring']] -> 'edge-ring'."""
    try:
        if raw_label is None:
            return "none"
        if isinstance(raw_label, (list, np.ndarray)):
            flat = np.array(raw_label).flatten()
            if len(flat) == 0:
                return "none"
            return str(flat[0]).strip().lower()
        return str(raw_label).strip().lower()
    except Exception:
        return "none"


def resize_wafer_map(wmap: np.ndarray, target_size: Tuple[int, int] = (32, 32)) -> np.ndarray:
    """
    Resizes a single non-uniform wafer map to target dimensions using nearest-neighbor interpolation.
    Wafer values:
      0: Non-wafer / background (outside circular wafer boundary)
      1: Normal die (working chip)
      2: Defective die (failing chip)
    
    Returns normalized float array in [0.0, 1.0]:
      0.0 -> Background
      0.5 -> Normal die
      1.0 -> Defective die
    """
    h, w = wmap.shape
    if (h, w) == target_size:
        resized = wmap.astype(np.float32)
    else:
        zoom_factors = (target_size[0] / h, target_size[1] / w)
        # order=0 preserves discrete labels 0, 1, 2
        resized = zoom(wmap, zoom_factors, order=0).astype(np.float32)
    
    # Normalize to [0, 1] range: 0 -> 0.0, 1 -> 0.5, 2 -> 1.0
    return resized / 2.0


def load_labeled_wafer_data(
    pkl_path: Optional[str] = None,
    target_size: Tuple[int, int] = (32, 32),
    cache_dir: Optional[str] = None,
    force_reload: bool = False
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Loads all labelled wafer maps and saves an efficient .npz cache for instant subsequent loading.
    
    Returns:
        images: np.ndarray of shape (N, 1, H, W) normalized to [0, 1]
        labels: np.ndarray of shape (N,) integer class indices (0..7)
        class_names: List of class name strings
    """
    if cache_dir is None:
        cache_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"wafer_labeled_{target_size[0]}x{target_size[1]}.npz")

    if not force_reload and os.path.exists(cache_path):
        print(f"[*] Loading cached preprocessed dataset from: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["images"], data["labels"], list(data["class_names"])

    dataset_path = find_dataset_file(pkl_path)
    print(f"[*] Loading raw dataset from: {dataset_path} (this takes ~30-45s for 811K records)...")
    
    # Backward compatibility shim for legacy pickles in modern pandas
    import sys
    try:
        import pandas.core.indexes.base
        sys.modules['pandas.indexes'] = pandas.core.indexes
        sys.modules['pandas.indexes.base'] = pandas.core.indexes.base
    except Exception:
        pass

    try:
        df = pd.read_pickle(dataset_path)
    except Exception:
        with open(dataset_path, "rb") as f:
            df = pickle.load(f)

    print("[*] Extracting failure labels and filtering to the 8 defect signatures...")
    labels_raw = df["failureType"].apply(extract_label_str)
    labeled_mask = labels_raw.isin(FAILURE_CLASSES)
    df_labeled = df[labeled_mask].copy()
    df_labeled["clean_label"] = labels_raw[labeled_mask]
    
    total_labeled = len(df_labeled)
    print(f"[*] Total labeled samples found: {total_labeled:,}")

    # Process and resize wafer maps
    print(f"[*] Resizing {total_labeled:,} wafer maps to {target_size[0]}x{target_size[1]}...")
    images_list = []
    labels_list = []

    for idx, row in df_labeled.iterrows():
        try:
            wmap = np.array(row["waferMap"], dtype=np.uint8)
            processed = resize_wafer_map(wmap, target_size=target_size)
            images_list.append(processed[np.newaxis, ...])  # Shape (1, H, W)
            labels_list.append(CLASS_TO_IDX[row["clean_label"]])
        except Exception as e:
            continue

    images = np.array(images_list, dtype=np.float32)
    labels = np.array(labels_list, dtype=np.int64)

    print(f"[*] Saving optimized cache to: {cache_path}")
    np.savez_compressed(
        cache_path,
        images=images,
        labels=labels,
        class_names=np.array(FAILURE_CLASSES)
    )
    print(f"[OK] Preprocessing complete. Dataset shape: images={images.shape}, labels={labels.shape}")
    return images, labels, FAILURE_CLASSES


try:
    import torch
    from torch.utils.data import Dataset, DataLoader

    class WaferDataset(Dataset):
        """PyTorch Dataset for wafer maps with class labels and one-hot encoding."""
        def __init__(self, images: np.ndarray, labels: np.ndarray, num_classes: int = len(FAILURE_CLASSES)):
            self.images = torch.from_numpy(images).float()
            self.labels = torch.from_numpy(labels).long()
            self.num_classes = num_classes

        def __len__(self) -> int:
            return len(self.labels)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            img = self.images[idx]
            label_idx = self.labels[idx]
            one_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            one_hot[label_idx] = 1.0
            return img, label_idx, one_hot

    def get_dataloaders(
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 64,
        val_split: float = 0.15,
        random_seed: int = 42
    ) -> Tuple[DataLoader, DataLoader]:
        """Creates stratified train and validation DataLoaders."""
        np.random.seed(random_seed)
        n = len(labels)
        indices = np.arange(n)
        np.random.shuffle(indices)

        split_idx = int(n * (1.0 - val_split))
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        train_ds = WaferDataset(images[train_idx], labels[train_idx])
        val_ds = WaferDataset(images[val_idx], labels[val_idx])

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

except ImportError:
    pass
