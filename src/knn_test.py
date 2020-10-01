import numpy as np
import faiss
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score





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

    
mnist = datasets.load_digits()

(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(mnist.data), mnist.target, test_size=0.25, random_state=42)

print("asd")
knn = FaissKNeighbors()
knn.fit(trainData,trainLabels)

predictions = knn.predict(testData)
print(accuracy_score(testLabels,predictions))
