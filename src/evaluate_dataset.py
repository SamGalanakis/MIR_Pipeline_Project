from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from sklearn.metrics import accuracy_score
from faiss_knn import FaissKNeighbors
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm
import cProfile
import numpy as np
import pprint


class Evaluater:
    def __init__(self,data_path,n_vertices_target,n_samples_query):


        self.df, *_ = process_dataset_for_knn(data_path,divide_distributions=False)
        self.df_numeric = self.df.select_dtypes(include=np.number)

        self.faiss_knn = FaissKNeighbors(self.df_numeric,metric='L2')
        self.faiss_knn.train()
        self.database_class_path = self.df[["classification","file_name"]]
        self.labels = {}
        self.n_samples_query = n_samples_query
        self.results = []
        
        #self.df["classification"].nunique
        self.models_per_class = Counter()
        for classification in self.database_class_path["classification"].values:
            self.models_per_class[classification] += 1


    def evaluate(self):
        for idx, (classification,model)  in tqdm(enumerate(self.database_class_path.values)):
            #Number of results is based on the number of shapes in a class 
            n_results = self.models_per_class[classification]
    
            _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)
            self.results.append((classification, indices.flatten()))


    def analysis(self):
        self.accuracy_per_class = defaultdict(list)
        self.overall_accuracy_per_class = 0
        self.overall_accuracy = 0
        wtf = 0

        for classification, indices in self.results:
            n_correct_returns = sum([1 if self.database_class_path["classification"].values[ind] == classification else 0 for ind in indices])
            local_acc = n_correct_returns / self.models_per_class[classification]

            self.accuracy_per_class[classification].append(local_acc)
            self.overall_accuracy += local_acc
            wtf += 1
        for classification, all_accuracy in self.accuracy_per_class.items():
            self.accuracy_per_class[classification] = sum(all_accuracy) / len(all_accuracy)   

        self.overall_accuracy = self.overall_accuracy /  wtf
      #  self.overall_accuracy_per_class =  self.accuracy_per_class.values() /  len(self.accuracy_per_class.values())



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
    


