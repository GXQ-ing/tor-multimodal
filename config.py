from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

@dataclass
class DataConfig:
    traffic_labels: Tuple[str, ...] = ("meek-web", "obfs4", "snowflake", "webtunnel")
    #traffic_labels: Tuple[str, ...] = ("User", "HiddenService", "RelayBridge", "NonTor")
    raw_dir: Path = Path("data/raw")
    interim_dir: Path = Path("data/interim")
    processed_dir: Path = Path("data/processed")
    max_sequence_length: int = 5000

@dataclass
class TrainingConfig:
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    save_dir: Path = Path("artifacts")


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainingConfig = field(default_factory=TrainingConfig)

def get_config() -> Config:
    return Config()

