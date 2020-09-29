import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from shape import Shape
from model_viewer import ModelViewer
data_path = Path("processed_data//testingDatamaker.csv")


df = pd.read_csv(data_path)


plt.hist(df["n_triangles"],bins="auto")
plt.show()

df_sorted = df.sort_values("n_triangles")

min_example = df_sorted.iloc[0,:]["file_name"].replace("\\","/")
max_example = df_sorted.iloc[-1,:]["file_name"].replace("\\","/")
mean_example =  df.iloc[(df['n_triangles']-df["n_triangles"].mean()).abs().argsort()[0]]["file_name"].replace("\\","/")
viewer = ModelViewer()

viewer.process(Path(min_example))
print('done')
