import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import replace
from phyelds.simulator import Simulator
from phyelds.simulator.exporter import ExporterConfig


def subareas_evaluation_csv_exporter(
        simulator: Simulator,
        time_delta: float,
        config: ExporterConfig,
        number_of_subareas: int,
        moving_node_ids: list[int],
        **kwargs
):
    aggregated_path = f'{config.output_directory}{config.experiment_name}.csv'
    if not os.path.exists(aggregated_path) or config.initial:
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        agg_df = init_aggregated_dataframe(number_of_subareas, aggregated_path)
        for node_id in moving_node_ids:
            individual_path = f'{config.output_directory}{config.experiment_name}_node-{node_id}.csv'
            init_individual_dataframe(number_of_subareas, individual_path)
    else:
        agg_df = pd.read_csv(aggregated_path)

    all_node_accuracies = []
    for node_id in moving_node_ids:
        node = _find_node(simulator, node_id)
        if node is None:
            continue
        node_accuracies = [node.data['outputs'][f'accuracy-area-{i}'] for i in range(number_of_subareas)]
        all_node_accuracies.append(node_accuracies)

        individual_path = f'{config.output_directory}{config.experiment_name}_node-{node_id}.csv'
        individual_df = pd.read_csv(individual_path)
        new_row = {f'Area-{i}-Accuracy': node_accuracies[i] for i in range(number_of_subareas)}
        individual_df = pd.concat([individual_df, pd.DataFrame([new_row])], ignore_index=True)
        individual_df.to_csv(individual_path, mode='w', index=False)

    if all_node_accuracies:
        accuracy_matrix = np.array(all_node_accuracies)
        means = accuracy_matrix.mean(axis=0)
        stds = accuracy_matrix.std(axis=0)

        new_row = {}
        for i in range(number_of_subareas):
            new_row[f'Area-{i}-Accuracy-Mean'] = means[i]
            new_row[f'Area-{i}-Accuracy-Std'] = stds[i]

        agg_df = pd.concat([agg_df, pd.DataFrame([new_row])], ignore_index=True)
        agg_df.to_csv(aggregated_path, mode='w', index=False)

    config = replace(config, initial=False)
    simulator.schedule_event(
        time_delta, subareas_evaluation_csv_exporter, simulator, time_delta, config, number_of_subareas, moving_node_ids, **kwargs
    )


def _find_node(simulator: Simulator, node_id: int):
    for node in simulator.environment.nodes.values():
        if node.id == node_id:
            return node
    return None


def init_aggregated_dataframe(number_of_subareas: int, file_path: str):
    columns = []
    for i in range(number_of_subareas):
        columns.append(f'Area-{i}-Accuracy-Mean')
        columns.append(f'Area-{i}-Accuracy-Std')
    try:
        os.remove(file_path)
    except OSError:
        pass
    return pd.DataFrame(columns=columns).astype('float64')


def init_individual_dataframe(number_of_subareas: int, file_path: str):
    columns = [f'Area-{i}-Accuracy' for i in range(number_of_subareas)]
    try:
        os.remove(file_path)
    except OSError:
        pass
    df = pd.DataFrame(columns=columns).astype('float64')
    df.to_csv(file_path, index=False)
    return df
