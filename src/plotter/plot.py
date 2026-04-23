import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data/experiment_seed-42_subareas-4_dataset-EMNIST_partitioning-Hard.csv")

df_long = df.reset_index().melt(
    id_vars="index",
    var_name="metric",
    value_name="value"
)


plt.figure(figsize=(10,6))
sns.lineplot(data=df_long, x="index", y="value", hue="metric")

plt.xlabel("Global Round")
plt.ylabel("Accuracy")
plt.title("")
plt.savefig('accuracy.pdf')