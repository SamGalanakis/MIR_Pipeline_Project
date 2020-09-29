from ast import parse
import matplotlib.pyplot as plt
from utils import parse_array_from_str
import pandas as pd
from pathlib import Path
import numpy as np
from shape import Shape
from model_viewer import ModelViewer
data_path = Path("processed_data//testingDatamaker.csv")


df = pd.read_csv(data_path)

# Make sure np arrays are read from strings, probably better way to do this tbh"

array_columns = ["bounding_box",
                "angle_three_vertices","barycenter_vertice", "two_vertices",
                "square_area_triangle", "cube_volume_tetrahedron" ]
                

df[array_columns]=df[array_columns].applymap(parse_array_from_str)
df[array_columns].drop("bounding_box",axis=1)=df[array_columns].drop("bounding_box",axis=1).applymap(lambda x: x/x.max())
plt.hist(df["n_triangles"],bins=25,range=(0,100000))
plt.xlabel("Number of triangles")
plt.ylabel("Occurences in dataset")
plt.show()

df_sorted = df.sort_values("n_triangles")

min_example = df_sorted.iloc[0,:]["file_name"].replace("\\","/")
max_example = df_sorted.iloc[-1,:]["file_name"].replace("\\","/")
mean_example =  df.iloc[(df['n_triangles']-df["n_triangles"].mean()).abs().argsort()[0]]["file_name"].replace("\\","/")
viewer = ModelViewer()

viewer.process(Path(min_example))
print('done')
