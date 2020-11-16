import numpy as np
import faiss   
from pathlib import Path
import random
import numpy as np


class FaissKNeighbors:
    '''Interface for knn queries via faiss library'''

    def __init__(self, dataset,  metric = "L2"):
     
        self.metric = metric
   
        self.dataset = dataset

    def train(self):
        
      

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

    def query_baseline(self,n_results):
        indices = random.choices(np.arange(0, len(self.dataset),1), k = n_results)
        distances = np.zeros(len(indices))

        return distances, indices



  


if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")

    knn = FaissKNeighbors(data_path)
    knn.train()
  