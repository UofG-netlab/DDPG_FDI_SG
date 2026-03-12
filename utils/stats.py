import pandas as pd

df = pd.read_csv("./output_data/res_trafo/loading_percent.csv", sep=";", index_col=0)

for trafo_id in df.columns:
    max_load = df[trafo_id].max()
    if max_load > 100:
        print(f"✅ Transformer {trafo_id} was overloaded: max loading = {max_load:.2f}%")
    else:
        print(f"⚠️ Transformer {trafo_id} was NOT overloaded: max loading = {max_load:.2f}%")
