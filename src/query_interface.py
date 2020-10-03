from process_data_for_knn import process_dataset_for_knn
from pathlib import Path

from faiss_knn import FaissKNeighbors
from file_reader import FileReader
from shape import Shape
from preprocessing import process
from feature_extractor import extract_features

def QueryInterface():
    def __init__(self,data_path,n_neighbours=5):
        self.single_numeric_columns, self.min_max_scaler, self.df = process_dataset_for_knn(data_path)
        self.faiss_knn = FaissKNeighbors(n_neighbours)
        self.reader = FileReader()


    def query(self,model_path):
        vertices, element_dict, info = self.reader().read(model_path)
        shape = Shape(vertices,element_dict,info)
        shape = process(shape)
        feature_dict = extract_features(shape)
        print("done")




if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")
    query_interface = QueryInterface(data_path)




