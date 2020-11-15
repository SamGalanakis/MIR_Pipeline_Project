import numpy as np
from pathlib import Path
import random
import numpy as np
from scipy.stats import wasserstein_distance
from process_data_for_knn import process_dataset_for_knn
from utils import  is_array_col
"""
    #Make new environmen with Python 3.7
    #Install CUDA 10.0 
    #RUN on new environment: conda install faiss-gpu cudatoolkit=10.0 -c pytorch 
"""
     

class CustomNeighbors:
    def __init__(self, processed_data_path):
       # self.index = None
        

        self.df , self.exclude, self.scaler, self.array_columns, self.array_lengths = process_dataset_for_knn(processed_data_path,divide_distributions=False)
        self.df_numeric = self.df.select_dtypes(np.number)
        self.histogram_col_dict = {}        
        for array_name in self.array_columns:
            self.histogram_col_dict[array_name]=[x for x in self.df.columns if is_array_col(self.array_columns,x)==array_name]
        

        self.histogram_col_ind = self.histogram_col_dict
        for key, value in self.histogram_col_dict.items():
            self.histogram_col_dict[key] = [self.df_numeric.columns.tolist().index(x) for x in value ]
        self.scalar_cols = [x for x in self.df_numeric.columns if not is_array_col(self.array_columns,x)]
        self.scalar_col_ind = [self.df_numeric.columns.tolist().index(x) for x in self.scalar_cols]
        self.metric = self.weighted_wasserstein
    def weighted_wasserstein(self,query,weights):
        query = query.flatten()
        distances = np.zeros((self.df.shape[0],len(self.array_columns)))
        for index,  array_name in enumerate(self.array_columns):
            relevant_indices  = self.histogram_col_ind[array_name]
            for row_ind in range(self.df_numeric.shape[0]):
                distances[row_ind,index]=wasserstein_distance(self.df_numeric.values[row_ind,relevant_indices],query[relevant_indices])
        scalar_distances = (self.df_numeric.values[:,self.scalar_col_ind] - query[self.scalar_col_ind])**2
        distances = np.concatenate([distances,scalar_distances],axis=1)
        
        distances = np.matmul(distances,weights.reshape(-1,1))
        return distances
        


        


    def query(self, query, n_results,weights = False):
        query = query.reshape((1,-1)).astype(np.float32)

        if isinstance(weights,bool):
            weights =  np.ones(14)
       
        distances = self.metric(query,weights)
        indices = np.argsort(distances,axis=0)[0:n_results]
        distances = distances.squeeze()[indices]


        return distances, indices

    def query_baseline(self,n_results):
        indices = random.choices(np.arange(0, len(self.dataset),1), k = n_results)
        distances = np.zeros(len(indices))

        return distances, indices



    


if __name__ == '__main__':
    data_path = Path("processed_data/data_coarse1_processed_10000_10000000.0.csv")

    knn = CustomNeighbors(data_path)
    
    knn.query()