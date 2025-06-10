import os
import shutil
import pandas as pd 

#     
for i, data in pd.read_csv("experiments/analysis/keys.csv").iterrows():
    ident = data["Identifier"]
    for d in os.listdir("/home/janneke/changepoint-gp/datasets/datasets"):
        if ident in d:
            shutil.copy(f"/home/janneke/changepoint-gp/datasets/datasets/{d}", "paper-example/data/synthetic")