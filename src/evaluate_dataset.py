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
        self.labels = {}
        self.results = []
        self.class_counts = self.df['classification'].value_counts()
        self.big_metrics = []

    def evaluate(self, baseline = False):
        self.baseline = baseline
        self.big_results = defaultdict(dict)
        for idx, (classification,model)  in enumerate(self.database_class_path.values):
            #Number of results is based on the number of shapes in a class 
            n_results = self.class_counts[classification]

            if self.data_path:
                _, indices = self.custom_knn.query(self.df_numeric.iloc[idx].values, n_results,self.weights)
                self.results.append((classification, indices.flatten()))


            elif self.baseline:
                _, indices = self.faiss_knn.query_baseline(n_results)
                self.results.append((classification, indices))
            else:
                _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)
                self.results.append((classification, indices.flatten()))

    def evaluate_big(self, baseline = False):
            self.baseline = baseline
            self.big_results = defaultdict(dict)
            for idx, (classification,model)  in enumerate(self.database_class_path.values):
                #Number of results is based on the number of shapes in a class 
                to_query = np.arange(5,205,5)
                results = []


                for n_results in to_query:
                    if self.baseline:
                        _, indices = self.faiss_knn.query_baseline( n_results)
                        results.append((classification, indices))

                    else:
                        _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)
                        results.append((classification, indices.flatten()))


                    self.big_results[n_results] = results
                    results = []

                n_results = self.class_counts[classification]

                if self.baseline:
                    _, indices = self.faiss_knn.query_baseline( n_results)
                    results.append((classification, indices))

                else:
                    _, indices = self.faiss_knn.query(self.df_numeric.iloc[idx].values, n_results)
                    results.append((classification, indices.flatten()))


                self.big_results[0] = results

    def analysis_big(self):   
        self.overall_accuracy_per_class = 0
        metrics = {"accuracy" : 0,"precision": 0, "recall" :0 ,"f1" :0}
        accuracy = []
        precision = []
        recall = []
        f1 = []
        temp = defaultdict(dict)
        for n_results, results in self.big_results.items():
            metrics = {"accuracy" : 0,"precision": 0, "recall" :0 ,"f1" :0}
            accuracy = []
            precision = []
            recall = []
            f1 = []

            for index, (classification, indices) in enumerate(results):
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
            temp[n_results] = metrics

        return temp

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
            if self.baseline:
                if recall[-1] == 0 or precision[-1] == 0:
                    f1.append(0)
                else:
                    f1.append(2 * ((recall[-1] * precision[-1]) / (recall[-1] + precision[-1])))

            else:
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
        del self.metrics["accuracy"]
        return self.metrics


def make_graphs(test,file_name,devided):
    
    sns.set_style('darkgrid')
   # del test.metrics["accuracy"]
    df = pd.DataFrame.from_dict(test.metrics,orient="index")
    df.columns = ["metric"]
   # df.drop(['accuracy'])
    ax = sns.barplot(x=df.index, y="metric", data=df)
    plt.xticks(
        rotation=45, 
        horizontalalignment='right',
        fontweight='light',
        fontsize='x-large'  
    )
    ax.tick_params(axis='x', labelsize=11)

    file_name = file_name.replace(".","")
    
    if file_name == "baseline":
        plt.title(f"Average metrics baseline")
    elif "coarse" and "processed" in file_name:
        plt.title(f"Average metrics for preprocessed coarse shapes")
    elif "processed" in file_name:
        plt.title(f"Average metrics for preprocessed shapes")
    else:
        plt.title(f"Average metrics for unprocessed shapes")

    plt.tight_layout()
    if devided:
        plt.savefig(fr"graphs/evaluations/devided_{file_name}_all_metrics",dpi=150)
    else:
        plt.savefig(fr"graphs/evaluations/{file_name}_all_metrics",dpi=150)
    plt.clf()

    for metric, classes in test.metrics_per_class.items():
        
        if metric == "accuracy":
            continue
        df = pd.DataFrame.from_dict(classes,orient="index")
        df.columns = [metric]
        ax = sns.barplot(x=df.index, y=metric, data=df)

        if file_name == "baseline":
            plt.title(f"{metric} per class metrics baseline")
        elif "processed" in file_name:
            plt.title(f"{metric} per class for preprocessed shapes")
        else:
            plt.title(f"{metric} per class for unprocessed shapes")

        if "coarse" in file_name:  
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, horizontalalignment='right', fontweight='light',fontsize='x-large' )
            ax.tick_params(axis='x', labelsize=7)
        else:
            ax = sns.barplot(x=df.index, y=metric, data=df)
            ax.set_xticklabels([], horizontalalignment='right', fontweight='light',fontsize='x-large' )
           # plt.xticks(
           #     rotation=45, 
           #     horizontalalignment='right',
           #     fontweight='light',
           #     fontsize='x-small'  
           # )
        ax.tick_params(axis='x', labelsize=7)


        plt.tight_layout()
        if devided:
            plt.savefig(fr"graphs/evaluations/devided_{file_name}_{metric}",dpi=150)
        else:
            plt.savefig(fr"graphs/evaluations/{file_name}_{metric}",dpi=150)
        plt.clf()

    print(f"{file_name} is done")

def make_accu_graphs(metrics,file_name,devided):
    file_name = file_name.replace(".","")

    sns.set_style('darkgrid')
   # del test.metrics["accuracy"]
    df = pd.DataFrame.from_dict(metrics,orient="index")
    #sns.scatterplot(data=df, x=df.index, y="precision",)
    plt.clf()

    
    labels = df.index.tolist()
    idx = labels.index(0)
    labels[idx] = 'class counts'
    ticks = np.arange(5,205,5)
    ticks = np.append(0,ticks)
    
    ax = sns.scatterplot(x = df.index, y="precision", data=df)
    ax = sns.scatterplot(x = df.index, y="recall", data=df)
    ax = sns.scatterplot(x = df.index, y="f1", data=df)

    plt.xticks([])
    labels = [ "class counts", 5,  10,  15,  20,  25,  30,  35,  40,  45,  50,  55,  60,  65,
        70,  75,  80,  85,  90,  95, 100, 105, 110, 115, 120, 125, 130,
       135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 195,
       200]
       
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
        plt.savefig(fr"graphs/evaluations/devided_{file_name}metrics_across_queries",dpi=150)
    else:
        plt.savefig(fr"graphs/evaluations/{file_name}metrics_across_queries",dpi=150)
    plt.clf()




if __name__ == '__main__':
    #data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
                
    divide_distributions=True
    cla_paths = ["data/benchmark/classification/v1/coarse1/coarse1Train.cla","data/benchmark/classification/v1/coarse2/coarse2Train.cla"]
    class_dicts = [get_princeton_classifications(cla_path) for cla_path in cla_paths]
    df, *_ = process_dataset_for_knn("processed_data/data_coarse1_processed_10000_10000000.0.csv",divide_distributions)
   
    
    for idx, class_dict in enumerate(class_dicts):
        df['classification'] = df.file_name.map(lambda x: class_dict[os.path.basename(x).split(".")[0].replace("m","")])

        # classifications = df['classification'].to_list()                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
        # test = Evaluator(df)
        # test.evaluate()
        # test.analysis()
        # make_graphs(test,f"data_coarse{idx+1}_processed",divide_distributions)

        test = Evaluator(df)
        test.evaluate_big()
        temp = test.analysis_big()
        make_accu_graphs(temp,f"data_coarse{idx+1}_processed",divide_distributions)

    #baseline_test = Evaluator(df)
    #baseline_test.evaluate(baseline = True)
    #baseline_test.analysis()
    #make_graphs(baseline_test,"baseline",True)