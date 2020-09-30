from ast import parse
import matplotlib.pyplot as plt
from scipy.stats import distributions
from utils import parse_array_from_str, model_feature_dist
import pandas as pd
from pathlib import Path
import numpy as np
from shape import Shape
from model_viewer import ModelViewer
from sklearn import preprocessing
from scipy.spatial.distance import euclidean,cosine
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer
data_path = Path("processed_data/dataTest.csv")


df = pd.read_csv(data_path,index_col=0)

# Make sure np arrays are read from strings, probably better way to do this tbh"

array_columns = ["bounding_box",
                "angle_three_vertices","barycenter_vertice", "two_vertices",
                "square_area_triangle", "cube_volume_tetrahedron" ]
                
distribution_columns = [ "angle_three_vertices","barycenter_vertice", "two_vertices",
                "square_area_triangle", "cube_volume_tetrahedron" ]

non_numeric_columns = ["file_name","id","classification"]

single_numeric_columns = list(set(df.columns)-set(non_numeric_columns+array_columns))
df[array_columns]=df[array_columns].applymap(parse_array_from_str)
df[distribution_columns]=df[distribution_columns].applymap(lambda x: x/x.sum()) 


for col in single_numeric_columns:
    df[col].fillna(df[col].median(),inplace=True)


#remove extreme outliers
df = df[(df[single_numeric_columns]<=df[single_numeric_columns].quantile(0.999)).all(axis=1)]
df = df[(df[single_numeric_columns]>=df[single_numeric_columns].quantile(0.001)).all(axis=1)]


x = df[single_numeric_columns].values

min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)

df[single_numeric_columns]= x_scaled




inputt = df.sample().iloc[0,:]
against = df


dist = model_feature_dist(inputt,against,single_numeric_columns,array_columns,euclidean)

sorted_index_by_dist= sorted(range(0,df.shape[0]),key = lambda x : dist[x],reverse=False)

viewer = ModelViewer()

for id in sorted_index_by_dist:
    path = df.iloc[id,:]["file_name"]
    viewer.process(Path(path))
plt.hist(df["n_triangles"],bins=25,range=(0,100000))
plt.xlabel("Number of triangles")
plt.ylabel("Occurences in dataset")
plt.show()

df_sorted = df.sort_values("n_triangles")

min_example = df_sorted.iloc[0,:]["file_name"].replace("\\","/")
max_example = df_sorted.iloc[-1,:]["file_name"].replace("\\","/")
mean_example =  df.iloc[(df['n_triangles']-df["n_triangles"].mean()).abs().argsort()[0]]["file_name"].replace("\\","/")


viewer.process(Path(max_example))
print('done')
