import subprocess
import multiprocessing
import itertools
import time
import sys

def get_experiment_configs():
    seed_start = 42
    seed_end = 49
    seeds = list(range(seed_start, seed_end))
    dataset_names = ['EMNIST']
    partitioning_methods = ['Hard']
    number_of_subareas = 4
    preferred_learning_device = "cpu"

    experiments = {
        'C2FL_merge': {
            'training_strategy': 'normal',
            'enable_replay': True,
            'distill_on_area_entry': False,
            'adaptable_area_weight': True,
            'area_weight': 0.4,
            'min_area_weight': 0.1,
            'max_area_weight': 0.3,
        },
        # 'C2FL_distillation': {
        #     'training_strategy': 'distillation',
        #     'enable_replay': True,
        #     'distill_on_area_entry': False,
        #     'alpha': 0.4,
        #     'min_current_alpha': 0.05,
        #     'max_current_alpha': 0.6,
        # },
        'FL_merge': {
            'training_strategy': 'normal',
            'enable_replay': False,
            'area_weight': 0.4,
            'distill_on_area_entry': False,
        },
        # 'FL_distillation': {
        #     'training_strategy': 'distillation',
        #     'enable_replay': False,
        #     'alpha': 0.3,
        #     'distill_on_area_entry': False,
        # },
        'CL': {
            'training_strategy': 'no_merge',
            'enable_replay': True,
            'distill_on_area_entry': False,
        },
        'Local': {
            'training_strategy': 'no_merge',
            'enable_replay': False,
            'distill_on_area_entry': False,
        }
    }

    configs = []
    for seed, experiment_name, dataset_name, partitioning_method in itertools.product(
        seeds, experiments.items(), dataset_names, partitioning_methods
    ):
        exp_name, params = experiment_name
        config = {
            'experiment_name': exp_name,
            'dataset_name': dataset_name,
            'partitioning_method': partitioning_method,
            'number_of_regions': number_of_subareas,
            'preferred_learning_device': preferred_learning_device,
            'training_strategy': params['training_strategy'],
            'distill_on_area_entry': params['distill_on_area_entry'],
            'enable_replay': params['enable_replay'],
            'adaptable_area_weight': params.get('adaptable_area_weight', True),
            'area_weight': params.get('area_weight', 0.3),
            'min_area_weight': params.get('min_area_weight', 0.1),
            'max_area_weight': params.get('max_area_weight', 0.3),
            'alpha': params.get('alpha', 0.5),
            'min_current_alpha': params.get('min_current_alpha', 0.1),
            'max_current_alpha': params.get('max_current_alpha', 0.9),
            'seed': seed,
        }
        configs.append(config)
    return configs


def build_command(config):
    cmd = [
        sys.executable, 'src/main.py',
        '--experiment_name', config['experiment_name'],
        '--dataset_name', config['dataset_name'],
        '--partitioning_method', config['partitioning_method'],
        '--number_of_regions', str(config['number_of_regions']),
        '--preferred_learning_device', config['preferred_learning_device'],
        '--training_strategy', config['training_strategy'],
        '--distill_on_area_entry', str(config['distill_on_area_entry']),
        '--enable_replay', str(config['enable_replay']),
        '--adaptable_area_weight', str(config['adaptable_area_weight']),
        '--area_weight', str(config['area_weight']),
        '--min_area_weight', str(config['min_area_weight']),
        '--max_area_weight', str(config['max_area_weight']),
        '--alpha', str(config['alpha']),
        '--min_current_alpha', str(config['min_current_alpha']),
        '--max_current_alpha', str(config['max_current_alpha']),
        '--seed', str(config['seed']),
    ]
    return cmd


def run_sequential(configs):
    for i, config in enumerate(configs):
        print(f'[{i+1}/{len(configs)}] Running {config["experiment_name"]} (seed={config["seed"]})')
        cmd = build_command(config)
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f'Experiment {config["experiment_name"]} (seed={config["seed"]}) failed with return code {result.returncode}')


def run_parallel(configs, max_workers=None):
    if max_workers is None:
        max_workers = multiprocessing.cpu_count()

    processes = []
    for i, config in enumerate(configs):
        cmd = build_command(config)

        while len(processes) >= max_workers:
            processes = [p for p in processes if p.poll() is None]
            if len(processes) >= max_workers:
                time.sleep(1)

        print(f'[{i+1}/{len(configs)}] Launching {config["experiment_name"]} (seed={config["seed"]})')
        processes.append(subprocess.Popen(cmd))

    for p in processes:
        p.wait()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Launch experiments')
    parser.add_argument('--mode', type=str, default='sequential', choices=['sequential', 'parallel'])
    parser.add_argument('--max_workers', type=int, default=None)
    args = parser.parse_args()

    configs = get_experiment_configs()
    print(f'Total experiments to run: {len(configs)}')

    if args.mode == 'sequential':
        run_sequential(configs)
    else:
        run_parallel(configs, args.max_workers)
