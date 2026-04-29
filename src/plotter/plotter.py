import glob 
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path 
import matplotlib.pyplot as plt

def load_all_data(experiments: list[str], filter_by_node = None) -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    data = dict()
    for experiment in experiments:
        by_node_filter = f'{filter_by_node}' if filter_by_node is not None else '*'
        files = glob.glob(f'data/*{experiment}*_node-{by_node_filter}.csv')
        dfs = []
        for file in files: 
            df = pd.read_csv(file)
            dfs.append(df)
        data[experiment] = mean_var_dataframe(dfs)
    return data


def mean_var_dataframe(dfs: list[pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    concat_df = pd.concat(dfs)
    mean_df = concat_df.groupby(level=0).mean()
    var_df = concat_df.groupby(level=0).var()
    return mean_df, var_df

def plot_accuracy_single_node(df, charts_path):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()
    df_mean, df_var = df

    colors = sns.color_palette('deep')
    
    for i, col in enumerate(df_mean.columns):
        mean_values = df_mean[col]
        var_values = df_var[col]
        x_area_change = 30 * (i+1) 
        axes[i].plot(mean_values, color=colors[i if i != 3 else 4], linewidth=4)
        lower_bound = mean_values - var_values
        upper_bound = mean_values + var_values
        axes[i].fill_between(df_mean.index, lower_bound, upper_bound, 
                             color=colors[i if i != 3 else 4], alpha=0.2, label='Varianza')
        axes[i].axvline(x=x_area_change, color=colors[3], linestyle='--', linewidth=4)                     
        area_id = col.split('-')[1]
        axes[i].set_title(f"Data from area {area_id}", fontsize=25, fontweight='bold')
        axes[i].set_xlabel("Global Round", fontsize=24)
        axes[i].set_ylabel("Accuracy", fontsize=24)
        axes[i].tick_params(axis='both', which='major', labelsize=22)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].set_ylim(0, 1)

    plt.suptitle("", fontsize=16, y=0.98)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.savefig(f'{charts_path}/moving-node-fl.pdf')
    plt.close()


def plot_total_accuracy(dict_experiments, charts_path):
    colors = sns.color_palette("deep", len(dict_experiments))
    
    plt.figure(figsize=(10, 6))
    
    for i, (name, (df_mean, df_var)) in enumerate(dict_experiments.items()):
        target_cols = [c for c in df_mean.columns if 'Accuracy' in c]
        total_mean = df_mean[target_cols].sum(axis=1)
        total_var = np.sqrt(df_var[target_cols].sum(axis=1))
        color = colors[i]
        
        plt.plot(total_mean.index, total_mean, label=name, color=color, linewidth=4)
        # plt.fill_between(total_mean.index, 
        #                  total_mean - total_var, 
        #                  total_mean + total_var, 
        #                  color=color, alpha=0.15)

    #plt.title("Somma dell'Accuratezza tra tutte le Aree", fontsize=14, fontweight='bold')
    plt.xlabel("Global Round", fontsize=25)
    plt.ylabel("Cumulative Accuracy", fontsize=25)
    
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(
        title="Experiment", 
        prop={'size': 20},           
        title_fontproperties={'size': 23} 
    )
    plt.tick_params(axis='both', labelsize=22)
    
    
    plt.subplots_adjust(left=0.12, bottom=0.15, right=0.95, top=0.90)
    #plt.tight_layout()
    plt.savefig(f'{charts_path}/comparison.pdf')

if __name__ == '__main__':
    experiments = ['C2FL_merge', 'FL_merge', 'CL', 'Local']
    all_data = load_all_data(experiments)
    node_zero_data = load_all_data(['FL_merge'], filter_by_node = 0)
    charts_path = 'charts'
    Path(charts_path).mkdir(exist_ok=True)

    plot_accuracy_single_node(node_zero_data['FL_merge'], charts_path)
    plot_total_accuracy(all_data, charts_path)