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
    def __init__(self, dataset,  metric = "L2"):
       # self.index = None
        self.metric = metric
   
        self.dataset = dataset

    def train(self):
        
        #to test it out
        #self.test_query = x_train[0]

        if self.metric == "L2":
            self.index = faiss.IndexFlatL2(self.dataset.shape[1])
            self.index.add(np.ascontiguousarray(self.dataset, dtype=np.float32))
        else:
            #COSINE SIMILARITY
            self.index = faiss.IndexFlatIP(self.dataset.shape[1])
            faiss.normalize_L2(self.dataset)
            self.index.add(np.ascontiguousarray(self.dataset, dtype=np.float32))


    def query(self, query, n_results):
        query = query.reshape((1,-1)).astype(np.float32)

        if self.metric == "L2":
            distances, indices = self.index.search(query,k=n_results)
        
        else:
            faiss.normalize_L2(query)
            distances, indices = self.index.search(query, k=n_results)


        return distances, indices


if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")

    knn = FaissKNeighbors(data_path)
    knn.train()
    #knn.query("asd",3)