import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import os
import re

csv_files = glob.glob("data/*.csv")

def get_exp_name(file):
    return file.split('42_')[-1].split('.')[0]

all_dfs = []

# 1. Individual plots
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


# 2. All experiments combined
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

# 2.1 All experiments combined - Average Accuracy
print("Generating combined average plot...")
avg_df = combined_df.groupby(['index', 'experiment'])['value'].mean().reset_index()
plt.figure(figsize=(14,8))
sns.lineplot(data=avg_df, x="index", y="value", hue="experiment")
plt.xlabel("Global Round")
plt.ylabel("Average Accuracy")
plt.title("All Experiments Combined - Average Accuracy")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig('accuracy_combined_all_average.pdf')
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
        sns.lineplot(data=pair_df, x="index", y="value", hue="metric", style="experiment")
        plt.xlabel("Global Round")
        plt.ylabel("Accuracy")
        plt.title(f"Comparison: {p1} vs {p2}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'accuracy_pair_{p1}_vs_{p2}.pdf')
        plt.close()

