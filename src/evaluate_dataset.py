from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from sklearn.metrics import accuracy_score
from faiss_knn import FaissKNeighbors
from collections import defaultdict
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import cProfile
import numpy as np
import pprint


class Evaluater:
    def __init__(self,df_knn_processed):


        self.df = df_knn_processed
        self.df_numeric = self.df.select_dtypes(include=np.number)

        self.faiss_knn = FaissKNeighbors(self.df_numeric,metric='L2')
        self.faiss_knn.train()
        self.database_class_path = self.df[["classification","file_name"]]
        self.labels = {}
        self.results = []
        
        
        self.class_counts = self.df['classification'].value_counts()


    def evaluate(self):
        for idx, (classification,model)  in tqdm(enumerate(self.database_class_path.values)):
            #Number of results is based on the number of shapes in a class 
            n_results = self.class_counts[classification]
    
            _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)
            self.results.append((classification, indices.flatten()))


    def analysis(self):
        
        self.accuracy_per_class = defaultdict(list)
        self.overall_accuracy_per_class = 0
        self.overall_accuracy = 0
        

        for index, (classification, indices) in enumerate(self.results):
            n_correct_returns = sum([self.df["classification"][ind] == classification for ind in indices])
            local_acc = n_correct_returns / self.class_counts[classification]

            self.accuracy_per_class[classification].append(local_acc)
            self.overall_accuracy += local_acc
           
        for classification, all_accuracy in self.accuracy_per_class.items():
            self.accuracy_per_class[classification] = sum(all_accuracy) / len(all_accuracy)   

        self.overall_accuracy = self.overall_accuracy /  len(self.results)

        



if __name__ == '__main__':
    profiler = cProfile.Profile()
    data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
    n_vertices_target = 10000
    n_samples_query = 1e+6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    test = Evaluater(data_path,n_vertices_target,n_samples_query)
    test.evaluate()
    test.analysis()
    #profiler.dump_stats('query_profile_stats')
    print(test.accuracy_per_class)
    print(test.overall_accuracy)
    
    print()
    


