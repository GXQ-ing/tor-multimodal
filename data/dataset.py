"""
Used for loading preprocessed multimodal data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class ManifestEntry:
    sequence_path: Path
    image_path: Path
    label: str


def load_manifest(manifest_path: Path) -> List[ManifestEntry]:
    entries: List[ManifestEntry] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    for item in data:
        entries.append(
            ManifestEntry(
                sequence_path=manifest_path.parent / item["sequence_path"],
                image_path=manifest_path.parent / item["image_path"],
                label=item["label"],
            )
        )
    return entries


class LabelEncoder:
    """
    Simple label encoder to maintain mapping between string labels and integer indices.
    """

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        self._stoi: Dict[str, int] = {}
        self._itos: List[str] = []
        if labels:
            for label in labels:
                self.add(label)

    def add(self, label: str) -> int:
        if label not in self._stoi:
            self._stoi[label] = len(self._itos)
            self._itos.append(label)
        return self._stoi[label]

    def encode(self, label: str) -> int:
        return self.add(label)

    def decode(self, index: int) -> str:
        return self._itos[index]

    @property
    def num_classes(self) -> int:
        return len(self._itos)


class TorTrafficDataset(Dataset):
    """
    Multimodal Dataset for Tor Traffic Analysis.
    Loads paired sequence and image features for deep learning models.
    """

    def __init__(
        self,
        manifest_path: Path,
        label_encoder: Optional[LabelEncoder] = None,
        cache_in_memory: bool = False,
    ) -> None:
        self.manifest_path = manifest_path
        self.entries = load_manifest(manifest_path)
        # Use provided encoder or build a new one from the manifest labels
        self.encoder = label_encoder or LabelEncoder(
            labels=[entry.label for entry in self.entries]
        )
        self.cache_in_memory = cache_in_memory
        self._cache: List[Optional[Dict[str, torch.Tensor]]] = (
            [None] * len(self.entries) if cache_in_memory else []
        )

    def __len__(self) -> int:
        return len(self.entries)

    def _load_numpy(self, path: Path) -> np.ndarray:
        """Helper to load .npy or .npz files."""
        if path.suffix == ".npy":
            return np.load(path)
        if path.suffix == ".npz":
            with np.load(path) as data:
                if "arr_0" in data:
                    return data["arr_0"]
                raise KeyError(f"Default array key 'arr_0' not found in {path}")
        raise ValueError(f"Unsupported file format: {path}")

    def _load_item(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        sequence = self._load_numpy(entry.sequence_path)
        image = self._load_numpy(entry.image_path)
        sample = {
            "sequence": torch.from_numpy(sequence).float(),
            "image": torch.from_numpy(image).unsqueeze(0).float() / 255.0,
            "label": torch.tensor(self.encoder.encode(entry.label), dtype=torch.long),
        }
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.cache_in_memory:
            cached = self._cache[idx]
            if cached is None:
                cached = self._load_item(idx)
                self._cache[idx] = cached
            return cached
        return self._load_item(idx)

    @property
    def num_classes(self) -> int:
        return self.encoder.num_classes

