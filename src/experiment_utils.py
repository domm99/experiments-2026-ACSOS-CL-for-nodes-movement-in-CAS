def format_ratio_for_name(value: float) -> str:
    return f"{value}".replace(".", "p")


def make_experiment_suffix(
    seed: int,
    number_of_regions: int,
    dataset_name: str,
    partitioning_method: str,
    traveler_enabled: bool,
    traveler_holdout_ratio: float,
) -> str:
    return (
        f"seed-{seed}_subareas-{number_of_regions}_dataset-{dataset_name}"
        f"_partitioning-{partitioning_method}_traveler-{traveler_enabled}"
        f"_holdout-{format_ratio_for_name(traveler_holdout_ratio)}"
    )


def make_experiment_name(
    seed: int,
    number_of_regions: int,
    dataset_name: str,
    partitioning_method: str,
    traveler_enabled: bool,
    traveler_holdout_ratio: float,
) -> str:
    return "experiment_" + make_experiment_suffix(
        seed,
        number_of_regions,
        dataset_name,
        partitioning_method,
        traveler_enabled,
        traveler_holdout_ratio,
    )
