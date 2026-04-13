from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeVar

import torch
from torch.utils.data import Dataset, Subset

from src.traveler.types import TravelerAreaArtifact, StateDict
from src.traveler.utils import clone_state_dict

T = TypeVar("T", bound=Dataset)

SNAPSHOT_FORMAT_VERSION = 1
SNAPSHOT_KIND = "traveler_static_fl"
MANIFEST_FILE_NAME = "manifest.json"
CONSENSUS_MODELS_DIR = "consensus_models"
TRAVELER_TRAIN_INDICES_DIR = "traveler_train_indices"
TRAVELER_TEST_INDICES_DIR = "traveler_test_indices"


@dataclass(frozen=True)
class TravelerSnapshot:
    manifest: dict[str, Any]
    consensus_models: dict[int, dict[str, torch.Tensor]]
    traveler_train_indices: dict[int, list[int]]
    traveler_test_indices: dict[int, list[int]]
    snapshot_dir: Path


def _normalize_area_items(items: Mapping[int, T] | Sequence[T]) -> list[tuple[int, T]]:
    if isinstance(items, Mapping):
        return sorted(items.items(), key=lambda pair: pair[0])
    return list(enumerate(items))


def _normalize_state_items(
    items: Mapping[int, StateDict] | Sequence[StateDict],
) -> dict[int, StateDict]:
    if isinstance(items, Mapping):
        return dict(items)
    return {index: state for index, state in enumerate(items)}


def _resolve_dataset_indices(dataset: Dataset) -> tuple[Dataset, list[int]]:
    if isinstance(dataset, Subset):
        base_dataset, base_indices = _resolve_dataset_indices(dataset.dataset)
        resolved_indices = [base_indices[index] for index in dataset.indices]
        return base_dataset, resolved_indices

    return dataset, list(range(len(dataset)))


def _dataset_indices_record(dataset: Dataset, area_id: int) -> dict[str, Any]:
    base_dataset, indices = _resolve_dataset_indices(dataset)
    return {
        "area_id": area_id,
        "dataset_type": f"{dataset.__class__.__module__}.{dataset.__class__.__qualname__}",
        "base_dataset_type": (
            f"{base_dataset.__class__.__module__}.{base_dataset.__class__.__qualname__}"
        ),
        "indices": indices,
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _area_file_map(
    directory_name: str,
    area_ids: Sequence[int],
    extension: str,
) -> dict[str, str]:
    return {
        str(area_id): f"{directory_name}/area_{area_id}.{extension}"
        for area_id in area_ids
    }


def save_traveler_snapshot(
    snapshot_dir: str | Path,
    *,
    manifest: Mapping[str, Any],
    consensus_models_by_area: Mapping[int, StateDict] | Sequence[StateDict],
    traveler_train_data_by_area: Mapping[int, Dataset] | Sequence[Dataset],
    traveler_test_data_by_area: Mapping[int, Dataset] | Sequence[Dataset],
) -> Path:
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)

    train_items = _normalize_area_items(traveler_train_data_by_area)
    test_items = _normalize_area_items(traveler_test_data_by_area)
    consensus_items = _normalize_state_items(consensus_models_by_area)

    train_map = dict(train_items)
    test_map = dict(test_items)
    area_ids = sorted(set(train_map) | set(test_map) | set(consensus_items))
    missing_train = [area_id for area_id in area_ids if area_id not in train_map]
    missing_test = [area_id for area_id in area_ids if area_id not in test_map]
    missing_models = [area_id for area_id in area_ids if area_id not in consensus_items]
    if missing_train or missing_test or missing_models:
        problems = []
        if missing_train:
            problems.append(f"missing traveler train data for areas {missing_train}")
        if missing_test:
            problems.append(f"missing traveler test data for areas {missing_test}")
        if missing_models:
            problems.append(f"missing consensus models for areas {missing_models}")
        raise ValueError("; ".join(problems))

    consensus_dir = snapshot_path / CONSENSUS_MODELS_DIR
    train_index_dir = snapshot_path / TRAVELER_TRAIN_INDICES_DIR
    test_index_dir = snapshot_path / TRAVELER_TEST_INDICES_DIR
    consensus_dir.mkdir(parents=True, exist_ok=True)
    train_index_dir.mkdir(parents=True, exist_ok=True)
    test_index_dir.mkdir(parents=True, exist_ok=True)

    consensus_file_map: dict[str, str] = {}
    train_index_file_map: dict[str, str] = {}
    test_index_file_map: dict[str, str] = {}

    for area_id in area_ids:
        consensus_file = consensus_dir / f"area_{area_id}.pt"
        torch.save(clone_state_dict(consensus_items[area_id]), consensus_file)
        consensus_file_map[str(area_id)] = f"{CONSENSUS_MODELS_DIR}/area_{area_id}.pt"

        train_index_record = _dataset_indices_record(train_map[area_id], area_id)
        train_index_file = train_index_dir / f"area_{area_id}.json"
        _write_json(train_index_file, train_index_record)
        train_index_file_map[str(area_id)] = (
            f"{TRAVELER_TRAIN_INDICES_DIR}/area_{area_id}.json"
        )

        test_index_record = _dataset_indices_record(test_map[area_id], area_id)
        test_index_file = test_index_dir / f"area_{area_id}.json"
        _write_json(test_index_file, test_index_record)
        test_index_file_map[str(area_id)] = (
            f"{TRAVELER_TEST_INDICES_DIR}/area_{area_id}.json"
        )

    manifest_payload = dict(manifest)
    manifest_payload.update(
        {
            "snapshot_format": SNAPSHOT_FORMAT_VERSION,
            "snapshot_kind": SNAPSHOT_KIND,
            "snapshot_created_utc": datetime.now(timezone.utc).isoformat(),
            "snapshot_area_ids": area_ids,
            "snapshot_files": {
                "consensus_models": consensus_file_map,
                "traveler_train_indices": train_index_file_map,
                "traveler_test_indices": test_index_file_map,
            },
        }
    )
    _write_json(snapshot_path / MANIFEST_FILE_NAME, manifest_payload)
    return snapshot_path


def load_traveler_snapshot(
    snapshot_dir: str | Path,
    *,
    map_location: str | torch.device | None = "cpu",
) -> TravelerSnapshot:
    snapshot_path = Path(snapshot_dir)
    manifest_path = snapshot_path / MANIFEST_FILE_NAME
    manifest = _read_json(manifest_path)

    if manifest.get("snapshot_format") != SNAPSHOT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported snapshot format: {manifest.get('snapshot_format')}"
        )
    if manifest.get("snapshot_kind") != SNAPSHOT_KIND:
        raise ValueError(f"Unsupported snapshot kind: {manifest.get('snapshot_kind')}")

    snapshot_files = manifest.get("snapshot_files")
    if not isinstance(snapshot_files, dict):
        raise ValueError("manifest.json is missing snapshot_files")

    consensus_files = snapshot_files.get("consensus_models", {})
    train_index_files = snapshot_files.get("traveler_train_indices", {})
    test_index_files = snapshot_files.get("traveler_test_indices", {})

    area_ids = manifest.get("snapshot_area_ids")
    if not isinstance(area_ids, list):
        raise ValueError("manifest.json is missing snapshot_area_ids")

    consensus_models: dict[int, dict[str, torch.Tensor]] = {}
    traveler_train_indices: dict[int, list[int]] = {}
    traveler_test_indices: dict[int, list[int]] = {}

    for area_id in area_ids:
        area_key = str(area_id)
        if area_key not in consensus_files:
            raise ValueError(f"Missing consensus model file for area {area_id}")
        if area_key not in train_index_files:
            raise ValueError(f"Missing traveler train index file for area {area_id}")
        if area_key not in test_index_files:
            raise ValueError(f"Missing traveler test index file for area {area_id}")

        consensus_model = torch.load(
            snapshot_path / consensus_files[area_key],
            map_location=map_location,
        )
        if not isinstance(consensus_model, Mapping):
            raise ValueError(
                f"Consensus model for area {area_id} is not a state dict"
            )
        consensus_models[area_id] = dict(consensus_model)

        train_index_record = _read_json(snapshot_path / train_index_files[area_key])
        test_index_record = _read_json(snapshot_path / test_index_files[area_key])
        traveler_train_indices[area_id] = list(train_index_record.get("indices", []))
        traveler_test_indices[area_id] = list(test_index_record.get("indices", []))

    return TravelerSnapshot(
        manifest=manifest,
        consensus_models=consensus_models,
        traveler_train_indices=traveler_train_indices,
        traveler_test_indices=traveler_test_indices,
        snapshot_dir=snapshot_path,
    )


def reconstruct_traveler_artifacts(
    snapshot: TravelerSnapshot,
    traveler_train_data_by_area: Mapping[int, Dataset] | Sequence[Dataset],
    traveler_test_data_by_area: Mapping[int, Dataset] | Sequence[Dataset],
) -> list[TravelerAreaArtifact]:
    train_items = dict(_normalize_area_items(traveler_train_data_by_area))
    test_items = dict(_normalize_area_items(traveler_test_data_by_area))

    artifacts: list[TravelerAreaArtifact] = []
    for area_id in sorted(snapshot.consensus_models):
        if area_id not in train_items:
            raise ValueError(f"Missing traveler train dataset for area {area_id}")
        if area_id not in test_items:
            raise ValueError(f"Missing traveler test dataset for area {area_id}")
        if area_id not in snapshot.traveler_train_indices:
            raise ValueError(f"Missing traveler train indices for area {area_id}")
        if area_id not in snapshot.traveler_test_indices:
            raise ValueError(f"Missing traveler test indices for area {area_id}")

        artifacts.append(
            TravelerAreaArtifact(
                area_id=area_id,
                train_data=Subset(
                    train_items[area_id],
                    snapshot.traveler_train_indices[area_id],
                ),
                test_data=Subset(
                    test_items[area_id],
                    snapshot.traveler_test_indices[area_id],
                ),
                consensus_model=snapshot.consensus_models[area_id],
            )
        )

    return artifacts
