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


class Evaluator:
    def __init__(self,df_knn_processed):


        self.df = df_knn_processed
        self.df_numeric = self.df.select_dtypes(include=np.number)

        self.faiss_knn = FaissKNeighbors(self.df_numeric,metric='L2')
        self.faiss_knn.train()
        self.database_class_path = self.df[["classification","file_name"]]
        self.labels = {}
        self.results = []
        
        self.class_counts = self.df['classification'].value_counts()

    def evaluate(self, baseline = False):
        for idx, (classification,model)  in tqdm(enumerate(self.database_class_path.values)):
            #Number of results is based on the number of shapes in a class 
            n_results = self.class_counts[classification]
    
            if baseline():
                _, indices = self.faiss_knn.query_baseline(self.df_numeric.iloc[idx].values, n_results)
            else:
                _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)
            
            self.results.append((classification, indices.flatten()))

        self.analysis()


    def analysis(self):   
        self.overall_accuracy_per_class = 0
        self.metrics = {"accuracy" : 0,"precision": 0, "recall" :0 ,"f1" :0}
        self.metrics_per_class =  {"accuracy" : defaultdict(list),"precision": defaultdict(list), "recall" :defaultdict(list) ,"f1" :defaultdict(list)}
        accuracy = []
        precision = []
        recall = []
        f1 = []

        for index, (classification, indices) in enumerate(self.results):
            true_positives = sum([self.df["classification"][ind] == classification for ind in indices])
            false_positives = len(indices) - true_positives
            false_negatives = self.class_counts[classification] - true_positives
            true_negatives =  len(self.df) - false_negatives - false_positives - true_positives

            accuracy.append((true_positives + true_negatives) / len(self.df)) 
            precision.append(true_positives / len(indices))
            recall.append(true_positives / self.class_counts[classification])
            f1.append(2 * ((recall[-1] * precision[-1]) / (recall[-1] + precision[-1])))

            self.metrics_per_class["accuracy"][classification].append(accuracy[-1])
            self.metrics_per_class["precision"][classification].append(precision[-1])
            self.metrics_per_class["recall"][classification].append(recall[-1])
            self.metrics_per_class["f1"][classification].append(f1[-1])
        
        self.metrics_per_class_weigthed = self.metrics_per_class.copy()
        self.metrics_weigthed = self.metrics.copy()

        for metric_key, classes  in self.metrics_per_class.items():
            for classification, metric in classes.items():
                self.metrics_per_class[metric_key][classification] = sum(metric) / len(metric)
                self.metrics_per_class_weigthed[metric_key][classification] = ((sum(metric) / len(metric)) * self.class_counts[classification]) / sum(self.class_counts)
            
        for metric_key, classes in self.metrics_per_class_weigthed.items():
            self.metrics_weigthed[metric_key] = sum(classes.values()) 


        self.metrics["accuracy"] = sum(accuracy) / len(self.results)
        self.metrics["precision"] = sum(precision) / len(self.results)
        self.metrics["recall"] = sum(recall) / len(self.results)
        self.metrics["f1"] = sum(f1) / len(self.results)


if __name__ == '__main__':
    data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
    df, *_ = process_dataset_for_knn(data_path,divide_distributions=False)
    classifications = df['classification'].to_list()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    test = Evaluator(df)
    test.evaluate()
    print(test.metrics)
    #profiler.dump_stats('query_profile_stats')
    print()
    


