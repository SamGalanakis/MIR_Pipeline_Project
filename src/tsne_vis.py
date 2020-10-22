from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd 
from process_data_for_knn import process_dataset_for_knn
from bioinfokit.visuz import cluster
import numpy as np
from sklearn.cluster import KMeans


dataset_path_processed = Path(r'processed_data\data_coarse1_processed_10000_1000000.0.csv')
df = pd.read_csv(dataset_path_processed)
df_knn, *_= process_dataset_for_knn(dataset_path_processed,divide_distributions=False)

tsne_em = TSNE(n_components=2, perplexity=50.0, n_iter=10000, verbose=1).fit_transform(df_knn.select_dtypes(np.number))
color_class = df_knn['classification'].to_numpy()
cluster.tsneplot(score=tsne_em,colorlist = color_class)

