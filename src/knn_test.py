import numpy as np
import faiss
from utils import parse_array_from_str
from sklearn import datasets
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from process_data_for_knn import process_dataset_for_knn
import pandas as pd


"""
    #Make new environmen with Python 3.7
    #Install CUDA 10.0 
    #RUN on new environment: conda install faiss-gpu cudatoolkit=10.0 -c pytorch 
"""


class FaissKNeighbors:
    def __init__(self, data_path):
        self.index = None
        self.data_path = Path(data_path)
        self.single_numeric_columns, self.min_max_scaler, self.df = process_dataset_for_knn(data_path)

    def train(self, ncentroids = 1024, niter = 20, verbose = True ):
        features = ['volume', 'surface_area','bounding_box_ratio', 'compactness', 'bounding_box_volume', 'diameter',
       'eccentricity', 'bounding_box_0', 'bounding_box_1', 'bounding_box_2',
       'bounding_box_3', 'bounding_box_4', 'bounding_box_5', 'bounding_box_6',
       'bounding_box_7', 'bounding_box_8', 'bounding_box_9', 'bounding_box_10',
       'bounding_box_11', 'bounding_box_12', 'bounding_box_13',
       'bounding_box_14', 'bounding_box_15', 'bounding_box_16',
       'bounding_box_17', 'bounding_box_18', 'bounding_box_19',
       'bounding_box_20', 'bounding_box_21', 'bounding_box_22',
       'bounding_box_23', 'angle_three_vertices_0', 'angle_three_vertices_1',
       'angle_three_vertices_2', 'angle_three_vertices_3',
       'angle_three_vertices_4', 'angle_three_vertices_5',
       'angle_three_vertices_6', 'angle_three_vertices_7',
       'angle_three_vertices_8', 'angle_three_vertices_9',
       'barycenter_vertice_0', 'barycenter_vertice_1', 'barycenter_vertice_2',
       'barycenter_vertice_3', 'barycenter_vertice_4', 'barycenter_vertice_5',
       'barycenter_vertice_6', 'barycenter_vertice_7', 'barycenter_vertice_8',
       'barycenter_vertice_9', 'two_vertices_0', 'two_vertices_1',
       'two_vertices_2', 'two_vertices_3', 'two_vertices_4', 'two_vertices_5',
       'two_vertices_6', 'two_vertices_7', 'two_vertices_8', 'two_vertices_9',
       'square_area_triangle_0', 'square_area_triangle_1',
       'square_area_triangle_2', 'square_area_triangle_3',
       'square_area_triangle_4', 'square_area_triangle_5',
       'square_area_triangle_6', 'square_area_triangle_7',
       'square_area_triangle_8', 'square_area_triangle_9',
       'cube_volume_tetrahedron_0', 'cube_volume_tetrahedron_1',
       'cube_volume_tetrahedron_2', 'cube_volume_tetrahedron_3',
       'cube_volume_tetrahedron_4', 'cube_volume_tetrahedron_5',
       'cube_volume_tetrahedron_6', 'cube_volume_tetrahedron_7',
       'cube_volume_tetrahedron_8', 'cube_volume_tetrahedron_9']

        x_train =  self.df[features].values

        d = x_train.shape[1]
        self.kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu = True)
        self.kmeans.train(x_train.astype(np.float32))

    def query(self, query, number_answers):
        query = self.min_max_scaler.fit_transform(query)




if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")
    a = process_dataset_for_knn(data_path)
    knn = FaissKNeighbors(data_path)
    knn.train()
    knn.query()