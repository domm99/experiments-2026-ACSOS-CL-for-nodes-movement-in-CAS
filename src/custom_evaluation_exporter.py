import os
import pandas as pd
from pathlib import Path
from dataclasses import replace
from phyelds.simulator import Simulator
from phyelds.simulator.exporter import ExporterConfig

def first(list, predicate):
    for item in list:
        if predicate(item):
            return item
    return None

def subareas_evaluation_csv_exporter(
        simulator: Simulator,
        time_delta: float,
        config: ExporterConfig,
        number_of_subareas: int,
        movable_node_id: int,
        **kwargs
):
    file_path = f'{config.output_directory}{config.experiment_name}.csv'
    if not os.path.exists(file_path) or config.initial:
        Path(config.output_directory).mkdir(parents=True, exist_ok=True)
        df = init_dataframe(number_of_subareas, file_path)
    else:
        df = pd.read_csv(file_path)
    nodes = simulator.environment.nodes.values()
    movable_node = first(nodes, lambda n: n.id == movable_node_id)
    new_row = {f'Area-{i}-Accuracy': movable_node.data['outputs'][f'accuracy-area-{i}'] for i in range(number_of_subareas)}
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(file_path, mode='w', index=False)
    config = replace(config, initial=False)
    simulator.schedule_event(
        time_delta, subareas_evaluation_csv_exporter, simulator, time_delta, config, number_of_subareas, movable_node_id, **kwargs
    )


def init_dataframe(number_of_subareas: int, file_path: str):
    columns = [f'Area-{i}-Accuracy' for i in range(number_of_subareas)]
    try:
        os.remove(file_path)
    except OSError:
        pass
    return pd.DataFrame(columns=columns).astype('float64')