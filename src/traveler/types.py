from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import torch
from torch.utils.data import Dataset


StateDict = Mapping[str, torch.Tensor]


@dataclass(frozen=True)
class TravelerAreaArtifact:
    area_id: int
    train_data: Dataset
    test_data: Dataset
    consensus_model: StateDict


@dataclass(frozen=True)
class TravelerStepResult:
    area_id: int
    current_accuracy: float
    current_loss: float
    cumulative_accuracy: float
    cumulative_loss: float
    replay_samples: int


@dataclass(frozen=True)
class TravelerConfig:
    dataset_name: str
    device: str = "cpu"
    seed: int = 42
    verbose: bool = True
    check_consensus_models: bool = False
    replay_mem_size: int = 0
    train_mb_size: int = 128
    train_epochs: int = 2
    eval_mb_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class TravelerRunResult:
    results: Sequence[TravelerStepResult]
    final_model_state: dict[str, torch.Tensor]
    csv_path: Path | None = None
