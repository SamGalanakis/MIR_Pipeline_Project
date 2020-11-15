from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from sklearn.metrics import accuracy_score
from faiss_knn import FaissKNeighbors
from custom_knn import CustomNeighbors
from collections import defaultdict
from pathlib import Path
import pandas as pd
import cProfile
import numpy as np
from utils import get_princeton_classifications

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import seaborn as sns
import time


class Evaluator:
    def __init__(self,df_knn_processed,weights=False,data_path = False):
        self.data_path = data_path
        self.weights=weights
        self.df = df_knn_processed
        self.df_numeric = self.df.select_dtypes(include=np.number)
        if self.data_path:
            self.custom_knn = CustomNeighbors(data_path)
        self.faiss_knn = FaissKNeighbors(self.df_numeric,metric='L2')
        self.faiss_knn.train()
      
        self.database_class_path = self.df[["classification","file_name"]]
        self.class_counts = self.df['classification'].value_counts()

    def evaluate(self, baseline = False,max_query=205):
            self.baseline = baseline
            self.all_results = defaultdict(dict)
                #Number of results is based on the number of shapes in a class 
            to_query = np.arange(5,max_query+5,5)
            results = []

            for idx, (classification,model)  in enumerate(self.database_class_path.values):    
                if baseline:
                    _, indices = self.faiss_knn.query_baseline(max_query)
                    results.append((classification, indices))
                else:
                    _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, max_query)                
                    results.append((classification, indices.flatten()))

            for n_results in to_query:
                self.all_results[n_results] = [(re[0],re[1][0:n_results]) for re in results]
                
            results =  []
            for idx, (classification,model)  in enumerate(self.database_class_path.values):
                n_results = self.class_counts[classification]
                if baseline:
                    _, indices = self.faiss_knn.query_baseline(n_results)
                    results.append((classification, indices))
                else:
                    _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)                
                    results.append((classification, indices.flatten()))


            self.all_results[0] = results
            print("Done evaluate")

    def analysis(self):   
        metrics_per_querysize = defaultdict(dict)
        for n_results, results in self.all_results.items():
            metrics = {"accuracy" : 0,"precision": 0, "recall" :0 ,"f1" :0}
            accuracy = []
            precision = []
            recall = []
            f1 = []

            for classification, indices in results:
                true_positives = sum([self.df["classification"][ind] == classification for ind in indices])
                false_positives = len(indices) - true_positives
                false_negatives = self.class_counts[classification] - true_positives
                true_negatives =  len(self.df) - false_negatives - false_positives - true_positives

                accuracy.append((true_positives + true_negatives) / len(self.df)) 
                precision.append(true_positives / len(indices))
                recall.append(true_positives / self.class_counts[classification])
                if recall[-1] == 0 or precision[-1] == 0:
                    f1.append(0)
                else:
                    f1.append(2 * ((recall[-1] * precision[-1]) / (recall[-1] + precision[-1])))

            metrics["accuracy"] = sum(accuracy) / len(results)
            metrics["precision"] = sum(precision) / len(results)
            metrics["recall"] = sum(recall) / len(results)
            metrics["f1"] = sum(f1) / len(results)
            del metrics["accuracy"]
            metrics_per_querysize[n_results] = metrics

        print("Done analysis")
        return metrics_per_querysize

def make_graphs(metrics,file_name,devided):
    file_name = file_name.replace(".","")

    sns.set_style('darkgrid')
   # del test.metrics["accuracy"]
    df = pd.DataFrame.from_dict(metrics,orient="index")
    #sns.scatterplot(data=df, x=df.index, y="precision",)
    
    labels = df.index.tolist()
    idx = labels.index(0)
    labels[idx] = 'class counts'
    ticks = np.arange(5,55,5)
    ticks = np.append(0,ticks)
    
    ax = sns.scatterplot(x = df.index, y="precision", data=df)
    ax = sns.scatterplot(x = df.index, y="recall", data=df)
    ax = sns.scatterplot(x = df.index, y="f1", data=df)

    plt.xticks([])
    labels = ticks.tolist()
    labels[0] = 'class counts'
       
    ax.get_xaxis().set_ticks([])
    plt.xticks(ticks, labels=labels,
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-small'  
    )
    ax.tick_params(axis='x', labelsize=7)


    ax.set(xlabel='Results per query', ylabel='Metrics')

    ax.legend(["precision","recall","f1"])

    plt.title(f"Metrics across different results per query")

    plt.tight_layout()
    if devided:
        plt.savefig(fr"graphs/evaluations/adevided_{file_name}metrics_across_queries",dpi=150)
    else:
        plt.savefig(fr"graphs/evaluations/a{file_name}metrics_across_queries",dpi=150)
    plt.clf()


if __name__ == '__main__':
    #data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
                
    divide_distributions=True
    cla_paths = ["data/benchmark/classification/v1/coarse1/coarse1Train.cla","data/benchmark/classification/v1/coarse2/coarse2Train.cla"]
    class_dicts = [get_princeton_classifications(cla_path) for cla_path in cla_paths]
    df, *_ = process_dataset_for_knn("processed_data/data_coarse1_processed_10000_10000000.0.csv",divide_distributions)
    
    baseline_test = Evaluator(df)
    baseline_test.evaluate(baseline = True)
    temp = baseline_test.analysis()



   # profiler.run('database.create_database(base_name,n_samples=n_samples,apply_processing=apply_processing,n_vertices_target=n_vertices_target)')
   # profiler.dump_stats("query")