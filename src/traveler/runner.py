import copy
import csv
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import torch
from avalanche.benchmarks.scenarios.dataset_scenario import benchmark_from_datasets
from avalanche.benchmarks.utils import as_taskaware_classification_dataset
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.supervised.strategy_wrappers import Naive
from torch.utils.data import Dataset

from src.learning import initialize_model
from src.traveler.datasets import build_traveler_artifacts
from src.traveler.types import (
    TravelerAreaArtifact,
    TravelerConfig,
    TravelerRunResult,
    TravelerStepResult,
)
from src.traveler.utils import clone_state_dict, state_dicts_equal, traveler_log

def _build_strategy(
    dataset_name: str,
    state_dict,
    config: TravelerConfig,
    evaluator: EvaluationPlugin,
):
    model = initialize_model(dataset_name)
    model.load_state_dict(copy.deepcopy(state_dict))
    # SGD is the usual choice for CL...
    # ...but Adam is the one used for the standard FL part.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    replay_plugin = ReplayPlugin(mem_size=config.replay_mem_size)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No loggers specified, metrics will not be logged",
        )
        return Naive(
            model=model,
            optimizer=optimizer,
            train_mb_size=config.train_mb_size,
            train_epochs=config.train_epochs,
            eval_mb_size=config.eval_mb_size,
            device=config.device,
            plugins=[replay_plugin],
            evaluator=evaluator,
        )


def _build_evaluation_plugin() -> EvaluationPlugin:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No loggers specified, metrics will not be logged",
        )
        return EvaluationPlugin(
            accuracy_metrics(experience=True, stream=True),
            loss_metrics(experience=True, stream=True),
            loggers=[],
            strict_checks=False,
        )


def _extract_stream_metrics(metrics: Mapping[str, float]) -> tuple[float, float]:
    accuracy_key = "Top1_Acc_Stream/eval_phase/test_stream"
    loss_key = "Loss_Stream/eval_phase/test_stream"
    if accuracy_key not in metrics or loss_key not in metrics:
        raise KeyError(
            "Avalanche evaluation did not return expected stream metrics: "
            f"missing {accuracy_key!r} or {loss_key!r}"
        )
    return float(metrics[accuracy_key]), float(metrics[loss_key])


def _run_traveler_from_artifacts(
    artifacts: Sequence[TravelerAreaArtifact],
    config: TravelerConfig,
    csv_path: str | Path | None = None,
) -> TravelerRunResult:
    csv_output_path = Path(csv_path) if csv_path is not None else None
    ordered_artifacts = list(artifacts)

    if not ordered_artifacts:
        raise ValueError("artifacts must not be empty")

    benchmark = benchmark_from_datasets(
        train=[
            as_taskaware_classification_dataset(artifact.train_data)
            for artifact in ordered_artifacts
        ],
        test=[
            as_taskaware_classification_dataset(artifact.test_data)
            for artifact in ordered_artifacts
        ],
    )
    evaluator = _build_evaluation_plugin()
    strategy = _build_strategy(
        config.dataset_name,
        ordered_artifacts[0].consensus_model,
        config,
        evaluator,
    )
    results: list[TravelerStepResult] = []
    cumulative_seen_train_samples = 0
    test_experiences = list(benchmark.test_stream)
    traveler_log(
        config.verbose,
        f"starting Avalanche traveler over {len(ordered_artifacts)} areas"
    )

    for experience_index, (artifact, train_experience, test_experience) in enumerate(
        zip(
            ordered_artifacts,
            benchmark.train_stream,
            test_experiences,
        )
    ):
        current_samples = len(artifact.train_data)
        replay_before_current = cumulative_seen_train_samples
        traveler_log(
            config.verbose,
            (
                f"area {artifact.area_id}: training on {current_samples} current samples + "
                f"{replay_before_current} replay samples"
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r"Call to deprecated function update.*",
            )
            strategy.train(train_experience)

        current_metrics = strategy.eval([test_experience])
        current_accuracy, current_loss = _extract_stream_metrics(current_metrics)
        cumulative_metrics = strategy.eval(test_experiences[: experience_index + 1])
        cumulative_accuracy, cumulative_loss = _extract_stream_metrics(cumulative_metrics)
        cumulative_seen_train_samples += current_samples
        result = TravelerStepResult(
            area_id=artifact.area_id,
            current_accuracy=current_accuracy,
            current_loss=current_loss,
            cumulative_accuracy=cumulative_accuracy,
            cumulative_loss=cumulative_loss,
            replay_samples=cumulative_seen_train_samples,
        )
        results.append(result)
        traveler_log(
            config.verbose,
            (
                f"area {result.area_id}: experience_acc={result.current_accuracy:.4f}, "
                f"cumulative_acc={result.cumulative_accuracy:.4f}, "
                f"replay_samples={result.replay_samples}"
            ),
        )

    if csv_output_path is not None:
        export_traveler_results(results, csv_output_path)
        traveler_log(config.verbose, f"wrote metrics to {csv_output_path}")

    return TravelerRunResult(
        results=results,
        final_model_state=clone_state_dict(strategy.model.state_dict()),
        csv_path=csv_output_path,
    )


def _collect_consensus_models(
    simulator,
    device_data: Mapping[int, object],
    check_consensus_models: bool = False,
) -> dict[int, dict[str, torch.Tensor]]:
    consensus_models: dict[int, dict[str, torch.Tensor]] = {}
    representative_nodes: dict[int, int] = {}
    for node_id in sorted(device_data.keys()):
        node = simulator.environment.nodes[node_id]
        area_id = device_data[node_id].area_id
        final_area_model = node.data["outputs"]["final_area_model"]
        cloned_model = clone_state_dict(final_area_model)

        if area_id not in consensus_models:
            consensus_models[area_id] = cloned_model
            representative_nodes[area_id] = node_id
            continue

        if not check_consensus_models:
            continue

        reference_node_id = representative_nodes[area_id]
        same_model, difference = state_dicts_equal(
            consensus_models[area_id],
            cloned_model,
            left_name=f"area {area_id} node {reference_node_id}",
            right_name=f"area {area_id} node {node_id}",
        )
        if not same_model:
            raise ValueError(
                f"Area {area_id} does not have a unique final_area_model: "
                f"{difference}"
            )
    return consensus_models


def collect_consensus_models(
    simulator,
    device_data: Mapping[int, object],
    *,
    check_consensus_models: bool = False,
) -> dict[int, dict[str, torch.Tensor]]:
    return _collect_consensus_models(
        simulator,
        device_data,
        check_consensus_models=check_consensus_models,
    )


def run_traveler_from_artifacts(
    *,
    artifacts: Sequence[TravelerAreaArtifact],
    dataset_name: str,
    learning_device: str,
    seed: int,
    verbose: bool = True,
    csv_path: str | Path | None = None,
    check_consensus_models: bool = False,
) -> TravelerRunResult:
    replay_mem_size = sum(len(artifact.train_data) for artifact in artifacts)
    config = TravelerConfig(
        dataset_name=dataset_name,
        device=learning_device,
        seed=seed,
        verbose=verbose,
        check_consensus_models=check_consensus_models,
        replay_mem_size=replay_mem_size,
    )
    return _run_traveler_from_artifacts(artifacts, config, csv_path=csv_path)


def run_traveler(
    *,
    simulator,
    device_data,
    traveler_area_train_datasets,
    test_environment,
    dataset_name: str,
    learning_device: str,
    seed: int,
    experiment_name: str,
    verbose: bool = True,
    check_consensus_models: bool = False,
) -> TravelerRunResult:
    consensus_models = _collect_consensus_models(
        simulator,
        device_data,
        check_consensus_models=check_consensus_models,
    )

    # Note: here "training_data" is actually the test data!
    traveler_test_data = {
        area_id: region.training_data
        for area_id, region in enumerate(test_environment.regions)
    }
    artifacts = build_traveler_artifacts(
        traveler_area_train_datasets,
        traveler_test_data,
        consensus_models,
    )
    csv_path = Path("data") / f"traveler_{experiment_name}.csv"
    return run_traveler_from_artifacts(
        artifacts=artifacts,
        dataset_name=dataset_name,
        learning_device=learning_device,
        seed=seed,
        verbose=verbose,
        csv_path=csv_path,
        check_consensus_models=check_consensus_models,
    )


def export_traveler_results(
    results: Sequence[TravelerStepResult],
    csv_path: str | Path,
) -> Path:
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "area_id",
                "current_accuracy",
                "current_loss",
                "cumulative_accuracy",
                "cumulative_loss",
                "replay_samples",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "area_id": result.area_id,
                    "current_accuracy": result.current_accuracy,
                    "current_loss": result.current_loss,
                    "cumulative_accuracy": result.cumulative_accuracy,
                    "cumulative_loss": result.cumulative_loss,
                    "replay_samples": result.replay_samples,
                }
            )
    return path
