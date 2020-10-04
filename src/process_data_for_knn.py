from ast import parse
import matplotlib.pyplot as plt
from scipy.stats import distributions
from utils import parse_array_from_str, model_feature_dist, is_array_col
import pandas as pd
from pathlib import Path
import numpy as np
from shape import Shape
from model_viewer import ModelViewer
from sklearn import preprocessing
from scipy.spatial.distance import euclidean,cosine
from scipy.stats import wasserstein_distance
from sklearn.impute import SimpleImputer




def process_dataset_for_knn(data_path):
    data_path = Path(data_path)
    df = pd.read_csv(data_path,index_col=0)

    # Make sure np arrays are read from strings, probably better way to do this tbh"

    array_columns = ["bounding_box",
                    "angle_three_vertices","barycenter_vertice", "two_vertices",
                    "square_area_triangle", "cube_volume_tetrahedron" ]


 
    



    #Get array lengths, replace...isdigit to make sure not to count things like bounding_box_volume as part of bounding_box array.
    array_lengths = [len([x for x in df.columns if is_array_col(array_columns,x)==y]) for y in array_columns]

    non_numeric_columns = ["file_name","classification"]


    single_numeric_columns = [ x for x in set(df.columns)-set(non_numeric_columns) if not is_array_col(array_columns,x)]


    assert len(single_numeric_columns) +len(non_numeric_columns) + sum(array_lengths) == df.shape[1], "Column counts may be incorrect!"

    for col in single_numeric_columns:
        df[col].fillna(df[col].median(),inplace=True)


    #remove extreme outliers
    df = df[(df[single_numeric_columns]<=df[single_numeric_columns].quantile(0.999)).all(axis=1)]
    df = df[(df[single_numeric_columns]>=df[single_numeric_columns].quantile(0.001)).all(axis=1)]

    # Do min max scaling
    x = df[single_numeric_columns].values

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    df[single_numeric_columns]= x_scaled

    #Divide array entries by lenght so they only contribute as a single entry in total when taking L2 norm

    for length , array_name in zip(array_lengths,array_columns):
        for col in df.columns:
            if is_array_col(array_columns,col)==array_name:
                df[col] = df[col]/np.sqrt(length)


    return single_numeric_columns, min_max_scaler, df


if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")
    a = process_dataset_for_knn(data_path)
   



# numeric_df =  df.select_dtypes(include=np.number)
# sample_numeric = sample[numeric_df.columns]
# dist = np.linalg.norm((numeric_df.values-sample_numeric.values).astype(np.float32),axis=1,ord=2)

