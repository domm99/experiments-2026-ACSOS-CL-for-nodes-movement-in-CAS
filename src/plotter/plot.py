import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

OUTPUT_DIR = "generated"
SEED_PLOTS_DIR = os.path.join(OUTPUT_DIR, "seed-plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEED_PLOTS_DIR, exist_ok=True)

CHANGE_AREA_EACH = 20

csv_files = glob.glob("data/*.csv")

def parse_filename(file):
    basename = os.path.basename(file).replace('.csv', '')
    match = re.match(r'experiment_seed-(\d+)_(.+?)(?:_node-\d+)?$', basename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

def is_individual_file(file):
    return bool(re.search(r'_node-\d+\.csv$', file))

def is_aggregated_file(file):
    return not is_individual_file(file) and file.endswith('.csv')

def get_number_of_areas(df):
    mean_cols = [c for c in df.columns if c.endswith('-Accuracy-Mean')]
    return len(mean_cols)

def compute_combined_accuracy(df, n_areas):
    rows = []
    for _, row in df.iterrows():
        means = []
        stds = []
        for area_id in range(n_areas):
            mean_col = f'Area-{area_id}-Accuracy-Mean'
            std_col = f'Area-{area_id}-Accuracy-Std'
            if mean_col in row and std_col in row:
                m = row[mean_col]
                s = row[std_col]
                if pd.notna(m) and m > 0:
                    means.append(m)
                    stds.append(s)
        if means:
            combined_mean = np.mean(means)
            combined_std = np.sqrt(np.mean(np.array(stds)**2)) if stds else 0
            rows.append({
                'index': row.name,
                'mean': combined_mean,
                'std': combined_std,
            })
    return pd.DataFrame(rows)

def compute_node_combined_accuracy(df):
    area_cols = [c for c in df.columns if re.match(r'Area-\d+-Accuracy$', c)]
    rows = []
    for _, row in df.iterrows():
        vals = [row[c] for c in area_cols if pd.notna(row[c]) and row[c] > 0]
        if vals:
            rows.append({
                'index': row.name,
                'combined': np.mean(vals),
            })
    return pd.DataFrame(rows)

def infer_home_area(df):
    area_cols = sorted([c for c in df.columns if re.match(r'Area-\d+-Accuracy$', c)],
                       key=lambda c: int(re.search(r'Area-(\d+)-Accuracy', c).group(1)))
    if len(area_cols) == 0 or df.empty:
        return 0
    first_row = df.iloc[0]
    for col in area_cols:
        if pd.notna(first_row[col]) and first_row[col] > 0:
            return int(re.search(r'Area-(\d+)-Accuracy', col).group(1))
    return 0

def realign_by_visit_order(df, home_area, n_areas=4):
    area_cols = sorted([c for c in df.columns if re.match(r'Area-\d+-Accuracy$', c)],
                       key=lambda c: int(re.search(r'Area-(\d+)-Accuracy', c).group(1)))
    if len(area_cols) == 0:
        return df
    result = df.reset_index(drop=True).copy()
    result['index'] = result.index
    for step in range(n_areas):
        actual_area = (home_area + step) % n_areas
        actual_col = f'Area-{actual_area}-Accuracy'
        visit_col = f'Visit-{step}-Accuracy'
        if actual_col in df.columns:
            result[visit_col] = df[actual_col].values
    return result

all_aggregated_dfs = []
all_individual_dfs = []
all_node_combined_dfs = []

for file in csv_files:
    print(f"Processing file: {file}")
    df = pd.read_csv(file)
    seed, exp_name = parse_filename(file)
    if exp_name is None:
        continue

    if is_individual_file(file):
        node_match = re.search(r'_node-(\d+)\.csv$', file)
        node_id = int(node_match.group(1)) if node_match else -1

        df_long = df.reset_index().melt(
            id_vars="index",
            var_name="metric",
            value_name="value"
        )
        df_long['experiment'] = exp_name
        df_long['seed'] = seed
        df_long['node_id'] = node_id
        all_individual_dfs.append(df_long)

        df_combined = compute_node_combined_accuracy(df)
        if not df_combined.empty:
            df_combined['experiment'] = exp_name
            df_combined['seed'] = seed
            df_combined['node_id'] = node_id
            all_node_combined_dfs.append(df_combined)
    elif is_aggregated_file(file):
        n_areas = get_number_of_areas(df)
        if n_areas == 0:
            df_long = df.reset_index().melt(
                id_vars="index",
                var_name="metric",
                value_name="value"
            )
            df_long['experiment'] = exp_name
            df_long['seed'] = seed
            all_aggregated_dfs.append(df_long)
        else:
            df_combined = compute_combined_accuracy(df, n_areas)
            if not df_combined.empty:
                df_combined['experiment'] = exp_name
                df_combined['seed'] = seed
                all_aggregated_dfs.append(df_combined)

if all_aggregated_dfs:
    combined_aggregated = pd.concat(all_aggregated_dfs)

    if 'mean' in combined_aggregated.columns:
        print("Generating per-seed combined accuracy plots (one subplot per experiment)...")
        all_experiments = sorted(combined_aggregated['experiment'].unique())
        n_exps = len(all_experiments)
        n_cols = 3
        n_rows = (n_exps + n_cols - 1) // n_cols

        for seed in sorted(combined_aggregated['seed'].unique()):
            seed_df = combined_aggregated[combined_aggregated['seed'] == seed]
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
            axes = axes.flatten() if n_rows > 1 else [axes]

            colors = sns.color_palette(n_colors=n_exps)
            exp_colors = dict(zip(all_experiments, colors))

            for idx, exp in enumerate(all_experiments):
                ax = axes[idx]
                exp_df = seed_df[seed_df['experiment'] == exp].copy()
                if exp_df.empty:
                    ax.set_visible(False)
                    continue
                exp_df['index'] = exp_df['index'].astype(int)
                exp_df = exp_df.sort_values('index')
                ax.fill_between(
                    exp_df['index'],
                    exp_df['mean'] - exp_df['std'],
                    exp_df['mean'] + exp_df['std'],
                    alpha=0.2,
                    color=exp_colors[exp],
                )
                ax.plot(exp_df['index'], exp_df['mean'], color=exp_colors[exp], linewidth=2)
                ax.set_title(exp, fontsize=12, fontweight='bold')
                ax.set_xlabel("Global Round")
                ax.set_ylabel("Combined Accuracy")
                ax.set_ylim(0, 1)
                ax.grid(True, alpha=0.3)

            for idx in range(n_exps, len(axes)):
                axes[idx].set_visible(False)

            fig.suptitle(f"Seed {seed} - Combined Accuracy per Experiment", fontsize=14, fontweight='bold', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(SEED_PLOTS_DIR, f'accuracy_combined_seed-{seed}.pdf'), bbox_inches='tight')
            plt.close()

        print("Generating combined average plot (all experiments, all seeds)...")
        plt.figure(figsize=(14, 8))
        colors = sns.color_palette(n_colors=len(combined_aggregated['experiment'].unique()))
        exp_colors = dict(zip(sorted(combined_aggregated['experiment'].unique()), colors))
        for exp in sorted(combined_aggregated['experiment'].unique()):
            exp_df = combined_aggregated[combined_aggregated['experiment'] == exp]
            exp_df = exp_df.copy()
            exp_df['index'] = exp_df['index'].astype(int)
            exp_df = exp_df.sort_values('index')
            plt.plot(exp_df['index'], exp_df['mean'], color=exp_colors[exp], label=exp, linewidth=2)
        plt.xlabel("Global Round")
        plt.ylabel("Combined Accuracy")
        plt.ylim(0, 1)
        plt.title("All Experiments Combined - Combined Accuracy (mean ± std)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_combined_all_average.pdf'))
        plt.close()

if all_node_combined_dfs:
    combined_nodes = pd.concat(all_node_combined_dfs)
    print("Generating per-seed per-node combined accuracy plots (aligned by visit order)...")
    all_experiments = sorted(combined_nodes['experiment'].unique())
    n_exps = len(all_experiments)
    n_cols = 3
    n_rows = (n_exps + n_cols - 1) // n_cols

    for seed in sorted(combined_nodes['seed'].unique()):
        seed_df = combined_nodes[combined_nodes['seed'] == seed]
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
        axes = axes.flatten() if n_rows > 1 else [axes]

        colors = sns.color_palette("tab10", n_colors=10)

        for idx, exp in enumerate(all_experiments):
            ax = axes[idx]
            exp_df = seed_df[seed_df['experiment'] == exp]
            if exp_df.empty:
                ax.set_visible(False)
                continue
            for node_id, node_df in exp_df.groupby('node_id'):
                node_df = node_df.copy()
                node_df['index'] = node_df['index'].astype(int)
                node_df = node_df.sort_values('index')
                ax.plot(node_df['index'], node_df['combined'], color=colors[node_id % 10], linewidth=1.5, alpha=0.7, label=f'Node {node_id}')
            ax.set_title(exp, fontsize=12, fontweight='bold')
            ax.set_xlabel("Global Round")
            ax.set_ylabel("Combined Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc='upper left')

        for idx in range(n_exps, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(f"Seed {seed} - Per-Node Combined Accuracy", fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(SEED_PLOTS_DIR, f'accuracy_per_node_seed-{seed}.pdf'), bbox_inches='tight')
        plt.close()

    print("Generating per-seed per-node accuracy plots...")
    all_node_files = [f for f in csv_files if is_individual_file(f)]

    for file in all_node_files:
        df = pd.read_csv(file)
        seed, exp_name = parse_filename(file)
        node_match = re.search(r'_node-(\d+)\.csv$', file)
        node_id = int(node_match.group(1)) if node_match else -1
        if exp_name is None:
            continue

        home_area = infer_home_area(df)
        df_aligned = realign_by_visit_order(df, home_area)

        visit_cols = sorted([c for c in df_aligned.columns if c.startswith('Visit-') and c.endswith('-Accuracy')])
        n_visits = len(visit_cols)
        n_cols = 2
        n_rows = (n_visits + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        axes = axes.flatten() if n_visits > 1 else [axes]

        colors = sns.color_palette("tab10", n_colors=n_visits)

        for i, col in enumerate(visit_cols):
            ax = axes[i]
            df_aligned['index'] = df_aligned.index.astype(int)
            ax.plot(df_aligned['index'], df_aligned[col], color=colors[i], linewidth=2)
            area_id = (home_area + i) % 4
            ax.set_title(f'Visit {i} (Area {area_id})', fontsize=11, fontweight='bold')
            ax.set_xlabel("Global Round")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)

        for i in range(n_visits, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle(f"{exp_name} - Node {node_id} (seed={seed}, home=Area {home_area})", fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(SEED_PLOTS_DIR, f'accuracy_node-{node_id}_{exp_name}_seed-{seed}.pdf'), bbox_inches='tight')
        plt.close()

    pairs = [
        ("FL_merge", "C2FL_merge"),
        ("Local", "CL"),
        ("FL_merge", "FL_distillation"),
        ("CL", "C2FL_distillation"),
        ("C2FL_merge", "CL"),
        ("Local", "FL_merge"),
    ]
    for p1, p2 in pairs:
        pair_df = combined_aggregated[combined_aggregated['experiment'].isin([p1, p2])]
        if len(pair_df['experiment'].unique()) == 2:
            print(f"Generating pair plot: {p1} vs {p2}")
            plt.figure(figsize=(12, 8))
            sns.lineplot(data=pair_df, x="index", y="mean", hue="experiment", errorbar="sd")
            plt.xlabel("Global Round")
            plt.ylabel("Combined Accuracy")
            plt.ylim(0, 1)
            plt.title(f"Comparison: {p1} vs {p2}")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'accuracy_pair_{p1}_vs_{p2}.pdf'))
            plt.close()
