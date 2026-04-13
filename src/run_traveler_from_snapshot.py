from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ProFed import download_dataset
from src.traveler import (
    SUPPORTED_TRAVELER_STRATEGIES,
    load_traveler_snapshot,
    reconstruct_traveler_artifacts,
    run_travelers_from_artifacts,
)


def get_current_learning_device() -> str:
    learning_device = "cpu"
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            learning_device = current_accelerator.type
    return learning_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the traveler phase from a saved static-FL snapshot.",
    )
    parser.add_argument("snapshot_dir", help="Path to the snapshot directory")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for the traveler run (default: auto-detect)",
    )
    parser.add_argument(
        "--csv-path",
        default=None,
        help="Optional CSV output path (default: data/traveler_<experiment_name>.csv)",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=SUPPORTED_TRAVELER_STRATEGIES,
        default=None,
        help=(
            "Traveler strategy to run. Repeat the flag to run multiple strategies. "
            "Default: run all supported strategies."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable traveler progress logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot = load_traveler_snapshot(args.snapshot_dir)

    dataset_name = snapshot.manifest["dataset_name"]
    seed = int(snapshot.manifest["seed"])
    experiment_name = snapshot.manifest["experiment_name"]
    learning_device = args.device or get_current_learning_device()

    train_data, test_data = download_dataset(dataset_name)

    traveler_train_data = {
        area_id: train_data
        for area_id in snapshot.traveler_train_indices
    }
    traveler_test_data = {
        area_id: test_data
        for area_id in snapshot.traveler_test_indices
    }
    artifacts = reconstruct_traveler_artifacts(
        snapshot,
        traveler_train_data,
        traveler_test_data,
    )

    csv_path = args.csv_path
    if csv_path is None:
        csv_path = Path("data") / f"traveler_{experiment_name}.csv"

    results = run_travelers_from_artifacts(
        artifacts=artifacts,
        dataset_name=dataset_name,
        learning_device=learning_device,
        seed=seed,
        strategy_names=args.strategy,
        verbose=not args.quiet,
        csv_path=csv_path,
    )
    for result in results.values():
        if result.csv_path is not None:
            print(result.csv_path)


if __name__ == "__main__":
    main()
