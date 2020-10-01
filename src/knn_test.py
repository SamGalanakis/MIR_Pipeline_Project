import numpy as np
import faiss
from utils import parse_array_from_str
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd


"""
    #Make new environmen with Python 3.7
    #Install CUDA 10.0 
    #RUN on new environment: conda install faiss-gpu cudatoolkit=10.0 -c pytorch 
"""
class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def predict(self, X):
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions


def test():
    mnist = datasets.load_digits()

    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

    knn = FaissKNeighbors()
    knn.fit(trainData,trainLabels)
    predictions = knn.predict(testData)
    print(accuracy_score(testLabels,predictions))

def knn_maker():

    data_path = Path("processed_data/dataTest.csv")
    
    df = pd.read_csv(data_path,index_col=0)
    
    # Make sure np arrays are read from strings, probably better way to do this tbh"
    
    array_columns = ["bounding_box",
                    "angle_three_vertices","barycenter_vertice", "two_vertices",
                    "square_area_triangle", "cube_volume_tetrahedron" ]
                    
    distribution_columns = [ "angle_three_vertices","barycenter_vertice", "two_vertices",
                    "square_area_triangle", "cube_volume_tetrahedron" ]
    
    non_numeric_columns = ["file_name","id","classification"]
    
    single_numeric_columns = list(set(df.columns)-set(non_numeric_columns+array_columns))
    df[array_columns]=df[array_columns].applymap(parse_array_from_str)
    df[distribution_columns]=df[distribution_columns].applymap(lambda x: x/x.sum()) 