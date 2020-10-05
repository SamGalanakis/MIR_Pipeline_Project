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
        array_indices_length_dict = {}
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
                query_vector[curr_index:curr_index+length] = feature_dict[array_name]
                array_indices_length_dict[curr_index]=length
                curr_index += length


      
        query_vector= self.min_max_scaler.transform(query_vector.reshape(1,-1))

        for ind, length in array_indices_length_dict.items():
            query_vector[0,ind:ind+length]= query_vector[0,ind:ind+length]/np.sqrt(length)

        distances, indices = self.faiss_knn.query(query_vector,n_results=5)
        
        resulting_paths = [self.df['file_name'].to_list()[ind] for ind in indices.flatten()]
        #Send results for visualization
        self.visualize_results(shape,resulting_paths,distances)

    def visualize_results(self,query_model,sorted_resulting_paths,distances):
       # query_model.view()
        for path in sorted_resulting_paths:
            match_path = Path(path)
            match_shape = Shape(*self.reader.read(match_path))
            resulting_classes = self.df[self.df.file_name.isin(sorted_resulting_paths)]['classification']
            print(f'Query resulted in results classified with following distances : {zip(resulting_classes,distances)}')

            match_shape.view()










if __name__ == '__main__':
    data_path = Path("processed_data/dataTest1000.csv")
    model_path = Path(r"data/benchmark/db/1/m102/m102.off")
    query_interface = QueryInterface(data_path)
    query_interface.query(model_path)
    print("done")




