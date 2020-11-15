from evaluate_dataset import Evaluator
from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from pathlib import Path
import numpy as np
import torch
from torch import  nn, optim

from tqdm import tqdm 
import matplotlib.pyplot as plt
import torch.nn.functional as F


data_path = "processed_data/data_coarse1_processed_10000_10000000.0.csv"




data_path = Path("processed_data/data_coarse1_processed_10000_10000000.0.csv")
df, *_ = process_dataset_for_knn(data_path,divide_distributions=True)
classifications = df['classification'].to_list()
classifications_unique = sorted(list(set(classifications)))
classifications_indexer = {x:classifications_unique.index(x) for x in classifications_unique}
classifications_numeric = [classifications_indexer[x] for x in classifications]
df_numeric = df.select_dtypes(include=np.number)











loss_overall=0
epoch_max = int(1e+6)


random_weights = np.random.rand(50*14).reshape(-1,14)
random_weights = random_weights/random_weights.sum(axis=1,keepdims=True)
evaluator = Evaluator(df,data_path = data_path,weights = np.ones(14))
f1_list = []
recall_list = []
for index in range(random_weights.shape[0]):
    weights = random_weights[index,:]
    
 
    evaluator.weights = weights
    evaluator.evaluate()
    metrics_dict = evaluator.analysis()
    precision = metrics_dict['precision']
    f1 = metrics_dict['precision']
    recall = metrics_dict['f1']
    f1_list.append(f1)
    recall_list.append(recall)

    
    print(f"Precision: {precision}, Recall: {recall}")
best_f1 = f1_list.index(max(f1_list))
best_recall = recall_list.index(max(recall_list))



    
    

        




