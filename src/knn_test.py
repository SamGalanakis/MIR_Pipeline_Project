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
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def query(self, query, number_answers):
        pass

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


def test():
    mnist = datasets.load_digits()

    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

    knn = FaissKNeighbors()
    knn.fit(trainData,trainLabels)
    predictions = knn.predict(testData)
    print(accuracy_score(testLabels,predictions))

def knn_maker():
    data_path = Path("processed_data/dataTest.csv")
    single_numeric_columns, min_max_scaler, df = process_dataset_for_knn(data_path)

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

    x_train, x_val = train_test_split(df[features].values, test_size=0.10)

    ncentroids = 1024
    niter = 20
    verbose = True
    d = x_train.shape[1]
    kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose, gpu = )
    kmeans.train(x_train.astype(np.float32))

knn_maker()