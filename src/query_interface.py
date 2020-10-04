from process_data_for_knn import process_dataset_for_knn
from pathlib import Path

from faiss_knn import FaissKNeighbors
from file_reader import FileReader
from shape import Shape
from preprocessing import process
from feature_extractor import extract_features
from utils import is_array_col
import numpy as np
class QueryInterface:
    def __init__(self,data_path):
        self.single_numeric_columns, self.min_max_scaler, self.df = process_dataset_for_knn(data_path)
        self.df_numeric = self.df.select_dtypes(include=np.number)
        self.faiss_knn = FaissKNeighbors(self.df_numeric,metric='L2')
        self.faiss_knn.train()
        self.reader = FileReader()


        self.array_col =  ["bounding_box",
                    "angle_three_vertices","barycenter_vertice", "two_vertices",
                    "square_area_triangle", "cube_volume_tetrahedron" ]


    def query(self,model_path):
        vertices, element_dict, info = self.reader.read(model_path)
        shape = Shape(vertices,element_dict,info)
        shape = process(shape,n_faces_target=5000)
        feature_dict = extract_features(shape)

        query_vector = np.zeros(shape=(self.df_numeric.shape[1],))
        curr_index = 0
        seen_arrays=[]
        for col in self.df_numeric.columns:
            if col in feature_dict.keys():
                query_vector[curr_index]= feature_dict[col]
                curr_index +=1
            else:
                array_name = is_array_col(self.array_col,col)
                if array_name in seen_arrays:
                    continue
                seen_arrays.append(array_name)
                length = feature_dict[array_name].size
                query_vector[curr_index:curr_index+length] = feature_dict[array_name]/np.sqrt(length)
                curr_index += length
        n_single =  len(self.single_numeric_columns)
        query_vector[0:n_single] = self.min_max_scaler.transform(query_vector[0:n_single].reshape(1,-1))

        distances, indices = self.faiss_knn.query(query_vector,n_results=5)
        



        print("done")




if __name__ == '__main__':
    data_path = Path("processed_data/dataTest.csv")
    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    query_interface = QueryInterface(data_path)
    query_interface.query(ant_path)
    print("done")




