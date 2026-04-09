from typing import Mapping, Sequence, TypeVar

import torch
from torch.utils.data import Dataset, Subset

from src.traveler.types import TravelerAreaArtifact

T = TypeVar("T", bound=Dataset)


def split_dataset_holdout(
    dataset: Dataset,
    holdout_ratio: float,
    seed: int,
    area_id: int = 0,
) -> tuple[Subset, Subset]:
    """Split a dataset deterministically into static and traveler shards."""
    if not 0.0 <= holdout_ratio <= 1.0:
        raise ValueError("holdout_ratio must be in [0, 1]")

    if isinstance(dataset, Subset):
        base_dataset = dataset.dataset
        available_indices = list(dataset.indices)
    else:
        base_dataset = dataset
        available_indices = list(range(len(dataset)))

    dataset_length = len(available_indices)
    if dataset_length == 0:
        return Subset(base_dataset, []), Subset(base_dataset, [])

    holdout_count = int(dataset_length * holdout_ratio)
    if holdout_ratio > 0.0 and holdout_count == 0:
        holdout_count = 1
    holdout_count = min(holdout_count, dataset_length)
    static_count = dataset_length - holdout_count

    generator = torch.Generator()
    generator.manual_seed(seed + area_id)
    shuffled_indices = torch.randperm(dataset_length, generator=generator).tolist()

    static_indices = [available_indices[index] for index in shuffled_indices[:static_count]]
    traveler_indices = [available_indices[index] for index in shuffled_indices[static_count:]]
    return Subset(base_dataset, static_indices), Subset(base_dataset, traveler_indices)


def _normalize_area_items(
    items: Mapping[int, T] | Sequence[T],
) -> list[tuple[int, T]]:
    if isinstance(items, Mapping):
        return sorted(items.items(), key=lambda pair: pair[0])
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:  # type: ignore[index]
        return list(items)  # type: ignore[return-value]
    return list(enumerate(items))


def _normalize_state_items(
    items: Mapping[int, Mapping[str, torch.Tensor]] | Sequence[Mapping[str, torch.Tensor]],
) -> dict[int, Mapping[str, torch.Tensor]]:
    if isinstance(items, Mapping):
        return dict(items)
    if items and isinstance(items[0], tuple) and len(items[0]) == 2:  # type: ignore[index]
        return {area_id: state for area_id, state in items}  # type: ignore[misc]
    return {index: state for index, state in enumerate(items)}


def build_traveler_artifacts(
    train_data_by_area: Mapping[int, Dataset] | Sequence[Dataset],
    test_data_by_area: Mapping[int, Dataset] | Sequence[Dataset],
    consensus_models_by_area: Mapping[int, Mapping[str, torch.Tensor]]
    | Sequence[Mapping[str, torch.Tensor]],
) -> list[TravelerAreaArtifact]:
    """Create ordered traveler artifacts from per-area datasets and models."""
    train_items = _normalize_area_items(train_data_by_area)
    test_items = _normalize_area_items(test_data_by_area)
    consensus_items = _normalize_state_items(consensus_models_by_area)

    train_map = dict(train_items)
    test_map = dict(test_items)
    area_ids = sorted(set(train_map) | set(test_map) | set(consensus_items))
    missing = [area_id for area_id in area_ids if area_id not in consensus_items]
    if missing:
        raise ValueError(f"Missing consensus model(s) for area(s): {missing}")

    artifacts: list[TravelerAreaArtifact] = []
    for area_id in area_ids:
        if area_id not in train_map:
            raise ValueError(f"Missing train dataset for area {area_id}")
        if area_id not in test_map:
            raise ValueError(f"Missing test dataset for area {area_id}")
        artifacts.append(
            TravelerAreaArtifact(
                area_id=area_id,
                train_data=train_map[area_id],
                test_data=test_map[area_id],
                consensus_model=consensus_items[area_id],
            )
        )
    return artifacts
