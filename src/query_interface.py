from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from pathlib import Path
from database_creator import data_dict_parser
from faiss_knn import FaissKNeighbors
from file_reader import FileReader
from shape import Shape
from preprocessing import process
from feature_extractor import extract_features
from utils import is_array_col
import numpy as np

class QueryInterface:
    def __init__(self,data_path):
        self.df, *self.sample_normalization_parameters = process_dataset_for_knn(data_path)
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
        shape = process(shape,n_faces_target=1000)
        feature_dict = extract_features(shape)
        feature_df = data_dict_parser(feature_dict)
        feature_df, _ = sample_normalizer(feature_df,*self.sample_normalization_parameters)
        feature_df_numeric = feature_df.select_dtypes(np.number)
        #Make sure columns identical and ordered
        assert list(feature_df_numeric.columns) == list(self.df_numeric.columns), "Column mismatch!"
        query_vector = feature_df_numeric.iloc[0,:].values.astype(np.float32)

        
        #manual = np.linalg.norm(self.df_numeric-query_vector,axis=1)

      

        distances, indices = self.faiss_knn.query(query_vector,n_results=20)
        
        resulting_paths = [self.df['file_name'].to_list()[ind] for ind in indices.flatten()]
        #Send results for visualization
        self.visualize_results(shape,resulting_paths,distances)

    def visualize_results(self,query_model,sorted_resulting_paths,distances):
       # query_model.view()
        print(f'Query resulted in results classified with following distances:')
        

        for path, dist in zip(sorted_resulting_paths,list(distances.flatten())):
  
            match_path = path
            match_shape = Shape(*self.reader.read(match_path))
            classification = self.df[self.df['file_name']==path]['classification'].values[0]
          #  resulting_classes = self.df[self.df.file_name.isin(sorted_resulting_paths)]['classification']
            
            
            print(f'{path} -- {classification} -- {dist}\n')

            #match_shape.view()

        print("Done")










if __name__ == '__main__':
    data_path = Path("processed_data/dataTest1000.csv")
    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    model_path = Path(r"data/benchmark/db/1/m102/m102.off")
    query_interface = QueryInterface(data_path)
    query_interface.query(ant_path)
    print("done")




