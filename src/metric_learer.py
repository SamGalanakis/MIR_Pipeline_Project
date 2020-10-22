from faiss_knn import  FaissKNeighbors
from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from pathlib import Path
import numpy as np


data_path = Path("processed_data/data_processed_10000_1000000.0.csv")
df, *_ = process_dataset_for_knn(data_path,divide_distributions=False)
df_numeric = df.df.select_dtypes(include=np.number)

faiss_knn = FaissKNeighbors(df_numeric,metric='L2')
faiss_knn.train()