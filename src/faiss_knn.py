import numpy as np
import faiss
from utils import parse_array_from_str
from sklearn import datasets
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from process_data_for_knn import process_dataset_for_knn
import pandas as pd
from shape import Shape
from file_reader import FileReader
from feature_extractor import extract_features

"""
    #Make new environmen with Python 3.7
    #Install CUDA 10.0 
    #RUN on new environment: conda install faiss-gpu cudatoolkit=10.0 -c pytorch 
"""


class FaissKNeighbors:
    def __init__(self, data_path,  metric = "L2"):
        self.index = None
        self.metric = None
        self.data_path = Path(data_path)
        _, _, self.df = process_dataset_for_knn(data_path)

    def train(self):
        if metric == "L2":
            x_train = self.df.select_dtypes(include=np.number).values
            self.index = faiss.IndexFlatL2(x_train.shape[1])
            self.index.add(np.ascontiguousarray(x_train, dtype=np.float32))
        else:
            #COSINE SIMILARITY
            self.index = faiss.IndexFlatIp(x_train.shape[1])
            self.index.add(np.ascontiguousarray(faiss.normalize_L2(x_train), dtype=np.float32))


    def query(self, query, number_answers):
        #to test it out
        #query = self.test_query.reshape(-1,84)
        if self.metric == "L2":
            distances, indices = self.index.search(query, k=number_answers)
        
        else:
            distances, indices = self.index.search(faiss.normalize_l2(query), k=number_answers)


        return indices


if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")

    knn = FaissKNeighbors(data_path)
    knn.train()
    knn.query("asd",3)