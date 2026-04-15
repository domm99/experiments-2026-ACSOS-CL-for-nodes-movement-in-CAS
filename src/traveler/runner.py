import copy
import csv
import warnings
from pathlib import Path
from typing import Mapping, Sequence

import torch
from avalanche.benchmarks.utils.classification_dataset import (
    _as_taskaware_supervised_classification_dataset,
)
from avalanche.evaluation.metrics import accuracy_metrics, loss_metrics
from avalanche.training.plugins import EvaluationPlugin
from avalanche.training.plugins import ReplayPlugin
from avalanche.training.supervised.der import DER
from avalanche.training.supervised.strategy_wrappers import LwF, Naive

from src.learning import initialize_model
from src.traveler.avalanche_patches import PatchedBiC
from src.traveler.benchmark import benchmark_from_datasets_mutable
from src.traveler.datasets import build_traveler_artifacts
from src.traveler.types import (
    TravelerAreaArtifact,
    TravelerConfig,
    TravelerRunResult,
    TravelerStepResult,
    TravelerStrategyName,
)
from src.traveler.utils import clone_state_dict, state_dicts_equal, traveler_log


SUPPORTED_TRAVELER_STRATEGIES: tuple[TravelerStrategyName, ...] = (
    "naive",
    "lwf",
    "replay",
    "lwf_replay",
    "bic",
    "derpp",
)


def _uses_replay(strategy_name: TravelerStrategyName) -> bool:
    return strategy_name in {"replay", "lwf_replay"}


def _uses_lwf(strategy_name: TravelerStrategyName) -> bool:
    return strategy_name in {"lwf", "lwf_replay"}


def _strategy_csv_path(
    csv_path: str | Path | None,
    strategy_name: TravelerStrategyName,
) -> Path | None:
    if csv_path is None:
        return None
    path = Path(csv_path)
    return path.with_name(f"{path.stem}_{strategy_name}{path.suffix}")


def build_traveler_config(
    *,
    artifacts: Sequence[TravelerAreaArtifact],
    dataset_name: str,
    learning_device: str,
    seed: int,
    strategy_name: TravelerStrategyName = "replay",
    verbose: bool = True,
    check_consensus_models: bool = False,
    overrides: Mapping[str, object] | None = None,
) -> TravelerConfig:
    if strategy_name not in SUPPORTED_TRAVELER_STRATEGIES:
        raise ValueError(
            f"Unknown traveler strategy {strategy_name!r}. "
            f"Supported values: {', '.join(SUPPORTED_TRAVELER_STRATEGIES)}"
        )

    config_kwargs: dict[str, object] = {
        "dataset_name": dataset_name,
        "strategy_name": strategy_name,
        "device": learning_device,
        "seed": seed,
        "verbose": verbose,
        "check_consensus_models": check_consensus_models,
        "replay_mem_size": sum(len(artifact.train_data) for artifact in artifacts),
    }
    if overrides is not None:
        config_kwargs.update(overrides)
    return TravelerConfig(**config_kwargs)


def _build_strategy(
    dataset_name: str,
    state_dict,
    config: TravelerConfig,
    evaluator: EvaluationPlugin,
):
    model = initialize_model(dataset_name)
    model.load_state_dict(copy.deepcopy(state_dict))
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    plugins = []
    if _uses_replay(config.strategy_name):
        plugins.append(ReplayPlugin(mem_size=config.replay_mem_size))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="No loggers specified, metrics will not be logged",
        )
        if config.strategy_name == "bic":
            return PatchedBiC(
                model=model,
                optimizer=optimizer,
                criterion=torch.nn.CrossEntropyLoss(),
                mem_size=config.replay_mem_size,
                val_percentage=config.bic_val_percentage,
                T=config.bic_temperature,
                stage_2_epochs=config.bic_stage_2_epochs,
                lamb=config.bic_lambda,
                lr=config.bic_lr,
                train_mb_size=config.train_mb_size,
                train_epochs=config.train_epochs,
                eval_mb_size=config.eval_mb_size,
                device=config.device,
                plugins=plugins,
                evaluator=evaluator,
            )
        if config.strategy_name == "derpp":
            return DER(
                model=model,
                optimizer=optimizer,
                criterion=torch.nn.CrossEntropyLoss(),
                mem_size=config.replay_mem_size,
                batch_size_mem=config.der_batch_size_mem,
                alpha=config.der_alpha,
                beta=config.der_beta,
                train_mb_size=config.train_mb_size,
                train_epochs=config.train_epochs,
                eval_mb_size=config.eval_mb_size,
                device=config.device,
                plugins=plugins,
                evaluator=evaluator,
            )
        if _uses_lwf(config.strategy_name):
            return LwF(
                model=model,
                optimizer=optimizer,
                criterion=torch.nn.CrossEntropyLoss(),
                alpha=config.lwf_alpha,
                temperature=config.lwf_temperature,
                train_mb_size=config.train_mb_size,
                train_epochs=config.train_epochs,
                eval_mb_size=config.eval_mb_size,
                device=config.device,
                plugins=plugins,
                evaluator=evaluator,
            )

        return Naive(
            model=model,
            optimizer=optimizer,
            train_mb_size=config.train_mb_size,
            train_epochs=config.train_epochs,
            eval_mb_size=config.eval_mb_size,
            device=config.device,
            plugins=plugins,
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

    benchmark = benchmark_from_datasets_mutable(
        train=[
            _as_taskaware_supervised_classification_dataset(artifact.train_data)
            for artifact in ordered_artifacts
        ],
        test=[
            _as_taskaware_supervised_classification_dataset(artifact.test_data)
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
        (
            f"starting Avalanche traveler strategy={config.strategy_name} "
            f"over {len(ordered_artifacts)} areas"
        ),
    )

    for experience_index, (artifact, train_experience, test_experience) in enumerate(
        zip(
            ordered_artifacts,
            benchmark.train_stream,
            test_experiences,
        )
    ):
        current_samples = len(artifact.train_data)
        replay_before_current = (
            min(cumulative_seen_train_samples, config.replay_mem_size)
            if _uses_replay(config.strategy_name)
            else 0
        )
        traveler_log(
            config.verbose,
            (
                f"strategy={config.strategy_name}, area {artifact.area_id}: "
                f"training on {current_samples} current samples + "
                f"{replay_before_current} replay samples"
            ),
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message=r"Call to deprecated function update.*",
            )
            strategy.train(train_experience, num_workers=8)

        current_metrics = strategy.eval([test_experience])
        current_accuracy, current_loss = _extract_stream_metrics(current_metrics)
        cumulative_metrics = strategy.eval(test_experiences[: experience_index + 1], num_workers=8)
        cumulative_accuracy, cumulative_loss = _extract_stream_metrics(cumulative_metrics)
        cumulative_seen_train_samples += current_samples
        replay_samples = (
            min(cumulative_seen_train_samples, config.replay_mem_size)
            if _uses_replay(config.strategy_name)
            else 0
        )
        result = TravelerStepResult(
            area_id=artifact.area_id,
            current_accuracy=current_accuracy,
            current_loss=current_loss,
            cumulative_accuracy=cumulative_accuracy,
            cumulative_loss=cumulative_loss,
            replay_samples=replay_samples,
        )
        results.append(result)
        traveler_log(
            config.verbose,
            (
                f"strategy={config.strategy_name}, area {result.area_id}: "
                f"experience_acc={result.current_accuracy:.4f}, "
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
    strategy_name: TravelerStrategyName = "replay",
    verbose: bool = True,
    csv_path: str | Path | None = None,
    check_consensus_models: bool = False,
) -> TravelerRunResult:
    config = build_traveler_config(
        artifacts=artifacts,
        dataset_name=dataset_name,
        learning_device=learning_device,
        seed=seed,
        strategy_name=strategy_name,
        verbose=verbose,
        check_consensus_models=check_consensus_models,
    )
    return _run_traveler_from_artifacts(
        artifacts,
        config,
        csv_path=_strategy_csv_path(csv_path, strategy_name),
    )


def run_traveler_from_artifacts_with_config(
    *,
    artifacts: Sequence[TravelerAreaArtifact],
    config: TravelerConfig,
    csv_path: str | Path | None = None,
) -> TravelerRunResult:
    if config.strategy_name not in SUPPORTED_TRAVELER_STRATEGIES:
        raise ValueError(
            f"Unknown traveler strategy {config.strategy_name!r}. "
            f"Supported values: {', '.join(SUPPORTED_TRAVELER_STRATEGIES)}"
        )
    return _run_traveler_from_artifacts(artifacts, config, csv_path=csv_path)


def run_travelers_from_artifacts(
    *,
    artifacts: Sequence[TravelerAreaArtifact],
    dataset_name: str,
    learning_device: str,
    seed: int,
    strategy_names: Sequence[str] | None = None,
    verbose: bool = True,
    csv_path: str | Path | None = None,
    check_consensus_models: bool = False,
) -> dict[TravelerStrategyName, TravelerRunResult]:
    selected_strategy_names = (
        list(SUPPORTED_TRAVELER_STRATEGIES)
        if strategy_names is None
        else list(dict.fromkeys(strategy_names))
    )
    if not selected_strategy_names:
        raise ValueError("At least one traveler strategy must be provided")
    results: dict[TravelerStrategyName, TravelerRunResult] = {}
    for strategy_name in selected_strategy_names:
        if strategy_name not in SUPPORTED_TRAVELER_STRATEGIES:
            raise ValueError(
                f"Unknown traveler strategy {strategy_name!r}. "
                f"Supported values: {', '.join(SUPPORTED_TRAVELER_STRATEGIES)}"
            )
        results[strategy_name] = run_traveler_from_artifacts(
            artifacts=artifacts,
            dataset_name=dataset_name,
            learning_device=learning_device,
            seed=seed,
            strategy_name=strategy_name,
            verbose=verbose,
            csv_path=csv_path,
            check_consensus_models=check_consensus_models,
        )
    return results


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
    strategy_name: TravelerStrategyName = "replay",
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
        strategy_name=strategy_name,
        verbose=verbose,
        csv_path=csv_path,
        check_consensus_models=check_consensus_models,
    )


def run_travelers(
    *,
    simulator,
    device_data,
    traveler_area_train_datasets,
    test_environment,
    dataset_name: str,
    learning_device: str,
    seed: int,
    experiment_name: str,
    strategy_names: Sequence[str] | None = None,
    verbose: bool = True,
    check_consensus_models: bool = False,
) -> dict[TravelerStrategyName, TravelerRunResult]:
    consensus_models = _collect_consensus_models(
        simulator,
        device_data,
        check_consensus_models=check_consensus_models,
    )

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
    return run_travelers_from_artifacts(
        artifacts=artifacts,
        dataset_name=dataset_name,
        learning_device=learning_device,
        seed=seed,
        strategy_names=strategy_names,
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
