import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

csv_files = glob.glob("data/*.csv")

def get_exp_name(file):
    match = re.search(r'kind-(.*?)-distill-(.*?)-replay-(.*?)\.csv', file)
    if match:
        kind, distill, replay = match.groups()
        return f"{kind}_distill-{distill}_replay-{replay}"
    return os.path.basename(file)

all_dfs = []

# 1. Quello che c'era prima (singoli plot)
for file in csv_files:
    print(f"Processing individual file: {file}")

    df = pd.read_csv(file)

    df_long = df.reset_index().melt(
        id_vars="index",
        var_name="metric",
        value_name="value"
    )
    
    exp_name = get_exp_name(file)
    df_long['experiment'] = exp_name
    all_dfs.append(df_long)

    plt.figure(figsize=(10,6))
    sns.lineplot(data=df_long, x="index", y="value", hue="metric")

    plt.xlabel("Global Round")
    plt.ylabel("Accuracy")
    plt.title(exp_name)
    plt.savefig(f'accuracy_{file.split("/")[-1].split(".")[0]}.pdf')
    plt.close()

combined_df = pd.concat(all_dfs)

# 2. In un solo grafico (usando stili diversi)
print("Generating combined plot...")
plt.figure(figsize=(14,8))
sns.lineplot(data=combined_df, x="index", y="value", hue="metric", style="experiment")
plt.xlabel("Global Round")
plt.ylabel("Accuracy")
plt.title("All Experiments Combined")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('accuracy_combined_all.pdf')
plt.close()

# 3. Falli a coppie
pairs = [
    ("normal_distill-False_replay-False", "normal_distill-False_replay-True"),
    ("no_merge_distill-False_replay-False", "no_merge_distill-False_replay-True"),
    ("normal_distill-False_replay-False", "distillation_distill-False_replay-True"),
    ("no_merge_distill-False_replay-True", "distillation_distill-False_replay-True"),
    ("normal_distill-False_replay-True", "normal_distill-False_replay-False"),
    ("normal_distill-False_replay-True", "no_merge_distill-False_replay-True"),
]

for p1, p2 in pairs:
    pair_df = combined_df[combined_df['experiment'].isin([p1, p2])]
    if len(pair_df['experiment'].unique()) == 2:
        print(f"Generating pair plot: {p1} vs {p2}")
        plt.figure(figsize=(12,8))
        sns.lineplot(data=pair_df, x="index", y="value", hue="metric", style="experiment")
        plt.xlabel("Global Round")
        plt.ylabel("Accuracy")
        plt.title(f"Comparison: {p1} vs {p2}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'accuracy_pair_{p1}_vs_{p2}.pdf')
        plt.close()

