from sklearn.metrics import accuracy_score
from query_interface import QueryInterface
from pathlib import Path
import pandas as pd
from collections import Counter, defaultdict
from tqdm import tqdm

class Evaluater:
    def __init__(self,data_path,n_vertices_target,n_samples_query):
        self.data_path = data_path
        self.query_interface = QueryInterface(data_path,divide_distributions=True,n_bins=10,n_vertices_target = n_vertices_target)
        self.database_class_path = pd.read_csv(data_path)[["classification","file_name"]]
        self.labels = {}
        self.models_per_class = Counter()
        self.n_samples_query = n_samples_query
        self.results = []
        self.accuracy_per_class = defaultdict(list)
        for classification in self.database_class_path["classification"].values:
            self.models_per_class[classification] += 1


        print()
    def evaluate(self):
        for classification,model  in tqdm(self.database_class_path.values):
            #Number of results is based on the number of shapes in a class 
            n_results = self.models_per_class[classification]

            self.query_interface.query(model.replace("\\","/"), self.n_samples_query, n_results)
            self.results.append((classification, self.query_interface.indices.flatten()))


    def analysis(self):
        

        for classification, indices in self.results:
            n_correct_returns = sum([1 if self.database_class_path["classification"].values[ind] == classification else 0 for ind in indices])
            self.accuracy_per_class[classification].append( n_correct_returns / self.models_per_class[classification])

     
        for classification, all_accuracy in self.accuracy_per_class.items():
            self.accuracy_per_class = sum(all_accuracy) / len(all_accuracy)     

if __name__ == '__main__':
    data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
    n_vertices_target = 10000
    n_samples_query = 1e+6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
    test = Evaluater(data_path,n_vertices_target,n_samples_query)
    test.evaluate()    
    test.analysis()
    print()
    


