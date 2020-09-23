import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from shape import Shape
from model_viewer import ModelViewer
data_path = Path("processed_data//data.csv")


df = pd.read_csv(data_path,index_col=0)


plt.hist(df["n_triangles"],bins="auto")
plt.show()

df_sorted = df.sort_values("n_triangles")

min_example = df_sorted.iloc[0,:]["file_name"]
max_example = df_sorted.iloc[-1,:]["file_name"]
mean_example =  df.iloc[(df['n_triangles']-df["n_triangles"].mean()).abs().argsort()[0]]["file_name"]
viewer = ModelViewer()

viewer.process(Path(min_example))
print('done')
