from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ProFed import download_dataset
from src.traveler import (
    SUPPORTED_TRAVELER_STRATEGIES,
    TravelerAreaArtifact,
    TravelerConfig,
    TravelerRunResult,
    TravelerStrategyName,
    build_traveler_config,
    load_traveler_snapshot,
    reconstruct_traveler_artifacts,
    run_traveler_from_artifacts_with_config,
)


DEFAULT_COMMON_GRID: dict[str, Sequence[object]] = {
    "learning_rate": [1e-4, 5e-4, 1e-3, 5e-3],
    "train_epochs": [1, 2, 4, 8, 16, 32],
}

DEFAULT_STRATEGY_GRIDS: dict[TravelerStrategyName, dict[str, Sequence[object]]] = {
    "naive": {},
    "lwf": {
        "lwf_alpha": [0.25, 0.5, 1.0, 2.0],
        "lwf_temperature": [1.0, 2.0, 4.0],
    },
    "replay": {},
    "lwf_replay": {
        "lwf_alpha": [0.5, 1.0, 2.0],
        "lwf_temperature": [2.0, 4.0],
    },
    "bic": {
        "bic_val_percentage": [0.05, 0.1],
        "bic_stage_2_epochs": [5, 20],
        "bic_lr": [0.05, 0.1, 0.05, 0.01],
    },
    "derpp": {
        "der_alpha": [0.1, 0.3, 0.5],
        "der_beta": [0.5, 1.0],
    },
}


def get_current_learning_device() -> str:
    learning_device = "cpu"
    if torch.accelerator.is_available():
        current_accelerator = torch.accelerator.current_accelerator()
        if current_accelerator is not None:
            learning_device = current_accelerator.type
    return learning_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run an extensive traveler hyperparameter grid search from a saved snapshot.",
    )
    parser.add_argument("snapshot_dir", help="Path to the snapshot directory")
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device for the traveler runs (default: auto-detect)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for detailed CSVs and the grid-search summary",
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=SUPPORTED_TRAVELER_STRATEGIES,
        default=None,
        help=(
            "Traveler strategy to sweep. Repeat the flag to sweep multiple strategies. "
            "Default: sweep all supported strategies."
        ),
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Optional cap on the total number of runs, useful for smoke testing",
    )
    parser.add_argument(
        "--parallelism",
        type=int,
        default=4,
        help="Number of parallel traveler runs to execute at once (default: 4)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging",
    )
    parser.add_argument(
        "--worker-spec",
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--worker-result",
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def _expand_grid(grid: Mapping[str, Sequence[object]]) -> list[dict[str, object]]:
    if not grid:
        return [{}]
    items = list(grid.items())
    keys = [key for key, _ in items]
    values = [list(values) for _, values in items]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def _format_value_for_name(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, "g").replace("-", "m").replace(".", "p")
    return str(value).replace("-", "m").replace(".", "p")


def _build_run_stem(
    strategy_name: TravelerStrategyName,
    run_index: int,
    params: Mapping[str, object],
) -> str:
    parts = [f"run-{run_index:04d}", f"strategy-{strategy_name}"]
    for key, value in params.items():
        parts.append(f"{key.replace('_', '-')}-{_format_value_for_name(value)}")
    return "_".join(parts)


def _resolve_overrides(
    raw_params: Mapping[str, object],
    total_train_samples: int,
) -> tuple[dict[str, object], dict[str, object]]:
    resolved = dict(raw_params)
    display_params = dict(raw_params)
    if "replay_mem_size" not in resolved:
        resolved["replay_mem_size"] = total_train_samples
    display_params["replay_mem_size"] = resolved["replay_mem_size"]
    return resolved, display_params


def _result_summary(
    *,
    strategy_name: TravelerStrategyName,
    run_index: int,
    config: TravelerConfig,
    result: TravelerRunResult,
    csv_path: Path,
    display_params: Mapping[str, object],
    total_train_samples: int,
) -> dict[str, object]:
    final_step = result.results[-1]
    mean_current_accuracy = (
        sum(step.current_accuracy for step in result.results) / len(result.results)
    )
    mean_cumulative_accuracy = (
        sum(step.cumulative_accuracy for step in result.results) / len(result.results)
    )
    mean_current_loss = (
        sum(step.current_loss for step in result.results) / len(result.results)
    )
    mean_cumulative_loss = (
        sum(step.cumulative_loss for step in result.results) / len(result.results)
    )

    row: dict[str, object] = {
        "run_index": run_index,
        "strategy": strategy_name,
        "csv_path": str(csv_path),
        "areas": len(result.results),
        "total_train_samples": total_train_samples,
        "final_current_accuracy": final_step.current_accuracy,
        "final_current_loss": final_step.current_loss,
        "final_cumulative_accuracy": final_step.cumulative_accuracy,
        "final_cumulative_loss": final_step.cumulative_loss,
        "mean_current_accuracy": mean_current_accuracy,
        "mean_current_loss": mean_current_loss,
        "mean_cumulative_accuracy": mean_cumulative_accuracy,
        "mean_cumulative_loss": mean_cumulative_loss,
        "selection_score": final_step.cumulative_accuracy,
    }
    row.update({f"config_{key}": value for key, value in asdict(config).items()})
    row.update({f"grid_{key}": value for key, value in display_params.items()})
    return row


def _write_summary(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_artifacts(
    snapshot_dir: str | Path,
) -> tuple[str, int, str, list[TravelerAreaArtifact]]:
    snapshot = load_traveler_snapshot(snapshot_dir)
    dataset_name = snapshot.manifest["dataset_name"]
    seed = int(snapshot.manifest["seed"])
    experiment_name = snapshot.manifest["experiment_name"]

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
    return dataset_name, seed, experiment_name, artifacts


def _build_run_specs(
    *,
    selected_strategies: Sequence[TravelerStrategyName],
    details_dir: Path,
    total_train_samples: int,
    max_runs: int | None,
) -> list[dict[str, object]]:
    run_specs: list[dict[str, object]] = []
    run_index = 0

    for strategy_name in selected_strategies:
        common_combinations = _expand_grid(DEFAULT_COMMON_GRID)
        specific_combinations = _expand_grid(DEFAULT_STRATEGY_GRIDS[strategy_name])
        for common_params in common_combinations:
            for specific_params in specific_combinations:
                if max_runs is not None and run_index >= max_runs:
                    return run_specs

                run_index += 1
                raw_params = dict(common_params)
                raw_params.update(specific_params)
                overrides, display_params = _resolve_overrides(
                    raw_params,
                    total_train_samples,
                )
                run_stem = _build_run_stem(strategy_name, run_index, display_params)
                csv_path = details_dir / strategy_name / f"{run_stem}.csv"
                run_specs.append(
                    {
                        "strategy_name": strategy_name,
                        "run_index": run_index,
                        "display_params": display_params,
                        "overrides": overrides,
                        "csv_path": str(csv_path),
                    }
                )

    return run_specs


def _run_single_spec(
    *,
    snapshot_dir: str | Path,
    learning_device: str,
    spec: Mapping[str, object],
) -> dict[str, object]:
    dataset_name, seed, _experiment_name, artifacts = _load_artifacts(snapshot_dir)
    total_train_samples = sum(len(artifact.train_data) for artifact in artifacts)
    strategy_name = spec["strategy_name"]
    run_index = spec["run_index"]
    display_params = spec["display_params"]
    overrides = spec["overrides"]
    csv_path = Path(spec["csv_path"])

    config = build_traveler_config(
        artifacts=artifacts,
        dataset_name=dataset_name,
        learning_device=learning_device,
        seed=seed,
        strategy_name=strategy_name,
        verbose=False,
        overrides=overrides,
    )
    result = run_traveler_from_artifacts_with_config(
        artifacts=artifacts,
        config=config,
        csv_path=csv_path,
    )
    return _result_summary(
        strategy_name=strategy_name,
        run_index=run_index,
        config=config,
        result=result,
        csv_path=csv_path,
        display_params=display_params,
        total_train_samples=total_train_samples,
    )


def _run_worker_mode(args: argparse.Namespace) -> int:
    if args.worker_spec is None or args.worker_result is None:
        raise ValueError("worker mode requires --worker-spec and --worker-result")

    spec_path = Path(args.worker_spec)
    result_path = Path(args.worker_result)
    with spec_path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)

    row = _run_single_spec(
        snapshot_dir=args.snapshot_dir,
        learning_device=args.device or get_current_learning_device(),
        spec=spec,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return 0


def main() -> None:
    args = parse_args()
    if args.worker_spec is not None or args.worker_result is not None:
        raise SystemExit(_run_worker_mode(args))

    dataset_name, seed, experiment_name, artifacts = _load_artifacts(args.snapshot_dir)
    _ = dataset_name, seed  # retained for symmetry with snapshot worker mode
    learning_device = args.device or get_current_learning_device()
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else Path("data") / "grid_search" / experiment_name
    )
    details_dir = output_dir / "details"
    summary_path = output_dir / "summary.csv"
    jobs_dir = output_dir / "_jobs"
    jobs_dir.mkdir(parents=True, exist_ok=True)

    selected_strategies = (
        list(SUPPORTED_TRAVELER_STRATEGIES)
        if args.strategy is None
        else list(dict.fromkeys(args.strategy))
    )

    total_train_samples = sum(len(artifact.train_data) for artifact in artifacts)
    run_specs = _build_run_specs(
        selected_strategies=selected_strategies,
        details_dir=details_dir,
        total_train_samples=total_train_samples,
        max_runs=args.max_runs,
    )

    for strategy_name in selected_strategies:
        common_combinations = _expand_grid(DEFAULT_COMMON_GRID)
        specific_combinations = _expand_grid(DEFAULT_STRATEGY_GRIDS[strategy_name])
        total_combinations = len(common_combinations) * len(specific_combinations)
        if not args.quiet:
            print(
                f"Sweeping {strategy_name}: {total_combinations} combinations "
                f"on snapshot {experiment_name}"
            )

    if not args.quiet:
        print(
            f"Running {len(run_specs)} traveler grid-search runs "
            f"with parallelism={args.parallelism}"
        )

    summary_rows: list[dict[str, object]] = []

    if args.parallelism <= 1:
        for spec in run_specs:
            if not args.quiet:
                print(
                    f"[{spec['run_index']}] {spec['strategy_name']} -> "
                    f"{Path(spec['csv_path']).name}"
                )
            summary_rows.append(
                _run_single_spec(
                    snapshot_dir=args.snapshot_dir,
                    learning_device=learning_device,
                    spec=spec,
                )
            )
            summary_rows.sort(key=lambda row: int(row["run_index"]))
            _write_summary(summary_rows, summary_path)
    else:
        pending_specs = list(run_specs)
        running: dict[int, tuple[subprocess.Popen[str], Mapping[str, object], Path, Path]] = {}
        script_path = Path(__file__).resolve()

        while pending_specs or running:
            while pending_specs and len(running) < args.parallelism:
                spec = pending_specs.pop(0)
                run_index = int(spec["run_index"])
                spec_path = jobs_dir / f"spec_{run_index:04d}.json"
                result_path = jobs_dir / f"result_{run_index:04d}.json"
                with spec_path.open("w", encoding="utf-8") as handle:
                    json.dump(spec, handle, indent=2, sort_keys=True)
                    handle.write("\n")

                command = [
                    sys.executable,
                    str(script_path),
                    str(args.snapshot_dir),
                    "--device",
                    learning_device,
                    "--worker-spec",
                    str(spec_path),
                    "--worker-result",
                    str(result_path),
                ]
                process = subprocess.Popen(
                    command,
                    cwd=str(PROJECT_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                running[run_index] = (process, spec, spec_path, result_path)
                if not args.quiet:
                    print(
                        f"[start {run_index}/{len(run_specs)}] "
                        f"{spec['strategy_name']} -> {Path(spec['csv_path']).name}"
                    )

            finished_run_indices: list[int] = []
            for run_index, (process, spec, spec_path, result_path) in running.items():
                return_code = process.poll()
                if return_code is None:
                    continue

                stdout, stderr = process.communicate()
                if return_code != 0:
                    raise RuntimeError(
                        f"Grid-search subprocess failed for run {run_index} "
                        f"({spec['strategy_name']}).\n"
                        f"stdout:\n{stdout}\n"
                        f"stderr:\n{stderr}"
                    )

                with result_path.open("r", encoding="utf-8") as handle:
                    row = json.load(handle)
                summary_rows.append(row)
                summary_rows.sort(key=lambda current_row: int(current_row["run_index"]))
                _write_summary(summary_rows, summary_path)
                finished_run_indices.append(run_index)
                if not args.quiet:
                    print(
                        f"[done {run_index}/{len(run_specs)}] "
                        f"{spec['strategy_name']} -> {Path(spec['csv_path']).name}"
                    )
                spec_path.unlink(missing_ok=True)
                result_path.unlink(missing_ok=True)

            for run_index in finished_run_indices:
                del running[run_index]

            if running and not finished_run_indices:
                time.sleep(0.2)

    if not args.quiet:
        print(summary_path)


if __name__ == "__main__":
    main()
