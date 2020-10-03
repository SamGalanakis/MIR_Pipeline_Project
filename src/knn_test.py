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

    def train(self, ncentroids = 10, niter = 20, verbose = True ):
        x_train = self.df.select_dtypes(include=np.number).values

        d = x_train.shape[1]
        self.kmeans = faiss.Kmeans(d, ncentroids, niter=niter, verbose=verbose)
        self.kmeans.train(np.ascontiguousarray(x_train, dtype=np.float32))

        print()

    def query(self, query, number_answers):
        query = self.min_max_scaler.fit_transform(query)
        self.kmeans.index.search()



if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")
    knn = FaissKNeighbors(data_path)
    knn.train()
   # knn.query()