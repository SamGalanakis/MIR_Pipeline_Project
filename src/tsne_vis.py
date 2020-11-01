from pandas.core.frame import DataFrame
from sklearn.manifold import TSNE
from pathlib import Path
import pandas as pd 
from process_data_for_knn import process_dataset_for_knn
from bioinfokit.visuz import cluster
import numpy as np
import os
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from utils import get_princeton_classifications
def make_tsne(data_path,save_path):

    # df = pd.read_csv(data_path)
    cla_path = 'data/benchmark/classification/v1/coarse1/coarse1Train.cla'
    # cla_path = 'data/benchmark/classification/v1/base/train.cla'
    classification_dict = get_princeton_classifications(cla_path)

    df_knn, *_= process_dataset_for_knn(data_path,divide_distributions=False)
    df_knn['classification'] = df_knn.file_name.map(lambda x: classification_dict[os.path.basename(x).split(".")[0].replace("m","")])
    tsne_em = TSNE(n_components=2, perplexity=40.0, n_iter=1000, verbose=1).fit_transform(df_knn.select_dtypes(np.number))
    unique_classes = sorted(list(set(df_knn['classification'])))
    classification_indexes = [unique_classes.index(x) for x in df_knn['classification']]
    hsv = plt.get_cmap('hsv')
    df=DataFrame()
    df['x_data']=tsne_em[:,0]
    df['y_data']=tsne_em[:,1]
    df['classification'] = df_knn['classification']
    df['name'] = df_knn['file_name'].map(lambda x : os.path.basename(x.replace('\\','/')).split('.')[0])
    df['file_name'] = df_knn['file_name']
    df.to_csv(save_path)
    colors = np.array([hsv(np.linspace(0, 1.0, len(unique_classes)))[x] for x in classification_indexes])
    plt.scatter(tsne_em[:,0],tsne_em[:,1],color=colors)
    plt.show()
    
    return tsne_em



if __name__ == '__main__':
    
    make_tsne(r'processed_data/data_coarse1_processed_10000_10000000.0.csv','processed_data/tsne_data.csv')

