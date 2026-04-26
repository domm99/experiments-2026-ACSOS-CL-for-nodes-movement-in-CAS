import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

OUTPUT_DIR = "generated"
SEED_PLOTS_DIR = os.path.join(OUTPUT_DIR, "seed-plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(SEED_PLOTS_DIR, exist_ok=True)

csv_files = glob.glob("data/*.csv")

def parse_filename(file):
    basename = os.path.basename(file).replace('.csv', '')
    match = re.match(r'experiment_seed-(\d+)_(.+)', basename)
    if match:
        return int(match.group(1)), match.group(2)
    return None, None

all_dfs = []

# 1. Individual plots
for file in csv_files:
    print(f"Processing individual file: {file}")

    df = pd.read_csv(file)
    seed, exp_name = parse_filename(file)
    if exp_name is None:
        continue

    df_long = df.reset_index().melt(
        id_vars="index",
        var_name="metric",
        value_name="value"
    )

    df_long['experiment'] = exp_name
    df_long['seed'] = seed
    all_dfs.append(df_long)

    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_long, x="index", y="value", hue="metric")

    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.title(f"{exp_name} (seed={seed})")
    plt.savefig(os.path.join(SEED_PLOTS_DIR, f'accuracy_{exp_name}_seed-{seed}.pdf'))
    plt.close()

combined_df = pd.concat(all_dfs)


# 2. All experiments combined - Average
print("Generating combined average plot...")
plt.figure(figsize=(14,8))
sns.lineplot(data=combined_df, x="index", y="value", hue="experiment", errorbar=None)
plt.xlabel("Global Round")
plt.ylabel("Average Accuracy")
plt.title("All Experiments Combined - Average Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'accuracy_combined_all_average.pdf'))
plt.close()

# 3. Only pairs of interest
pairs = [
    ("FL_merge", "C2FL_merge"),
    ("Local", "CL"),
    ("FL_merge", "FL_distillation"),
    ("CL", "C2FL_distillation"),
    ("C2FL_merge", "CL"),
    ("Local", "FL_merge"),
]

for p1, p2 in pairs:
    pair_df = combined_df[combined_df['experiment'].isin([p1, p2])]
    if len(pair_df['experiment'].unique()) == 2:
        print(f"Generating pair plot: {p1} vs {p2}")
        plt.figure(figsize=(12,8))
        sns.lineplot(data=pair_df, x="index", y="value", hue="experiment", errorbar=None)
        plt.xlabel("Global Round")
        plt.ylabel("Accuracy")
        plt.title(f"Comparison: {p1} vs {p2}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'accuracy_pair_{p1}_vs_{p2}.pdf'))
        plt.close()
