import numpy as np
import faiss   
from pathlib import Path
import random
import numpy as np


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
            distances, indices = self.index.search(query,k=int(n_results))
        
        else:
            faiss.normalize_L2(query)
            distances, indices = self.index.search(query, k=int(n_results))


        return distances, indices

    def query_baseline(self,query,n_results):
        indices = random.choices(np.arange(0, len(self.dataset),1), n_results)
        distances = np.zeros(len(indices))

        return distances, indices



    def query_range(self, query1, query2, n_results):
        #query = query.resha2pe((1,-1)).astype(np.float32)
        self.index.range_search()
        #return distances, indices


if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")

    knn = FaissKNeighbors(data_path)
    knn.train()
    #knn.query("asd",3)