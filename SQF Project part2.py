# %% read dataframe
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules


df = pd.read_pickle("data.pkl")


# %% converted physical force used columns to booleans
pfs = [col for col in df.columns if col.startswith("pf_")]
for col in pfs:
    df[col] = df[col] == "YES"


# %% manual one hot encoding for race / city
for val in df["race"].unique():
    df[f"race_{val}"] = df["race"] == val

for val in df["city"].unique():
    df[f"city_{val}"] = df["city"] == val

# %% convert inout (arrested inside or Outside) to boolean
df["inside"] = df["inout"] == "INSIDE"

# %% created 'armed" column
df["armed"] = (
    (df["contrabn"] == "YES")
    | (df["pistol"] == "YES")
    | (df["riflshot"] == "YES")
    | (df["asltweap"] == "YES")
    | (df["knifcuti"] == "YES")
    | (df["machgun"] == "YES")
    | (df["othrweap"] == "YES")
)

# %% selected columns for association rules mining
cols = [
    col
    for col in df.columns
    if col.startswith("pf_") or col.startswith("race_") or col.startswith("city_")
] + ["inside", "armed"]

# %% applied frequent itemset mining
frequent_itemsets = apriori(df[cols], min_support=0.01, use_colnames=True)
frequent_itemsets

# %% applied association rules mining
rules = association_rules(frequent_itemsets, min_threshold=0.3)
rules

# %% sorted rules by confidence
rules.sort_values("confidence", ascending=False)

# %% scatterplot support vs confidence
import seaborn as sns
import matplotlib.pyplot as plt

sns.scatterplot(x=rules["support"], y=rules["confidence"], alpha=0.5)
plt.xlabel("Support")
plt.ylabel("Confidence")
plt.title("Support vs Confidence")
plt.show()
# %%

