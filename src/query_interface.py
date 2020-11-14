from process_data_for_knn import process_dataset_for_knn,sample_normalizer
from pathlib import Path
from database_creator import data_dict_parser
from faiss_knn import FaissKNeighbors
from file_reader import read_model, write_model_as_ply
from shape import Shape
from preprocessing import process
from feature_extractor import extract_features
import numpy as np
import cProfile
import pandas as pd
from custom_knn import CustomNeighbors

class QueryInterface:
    def __init__(self,data_path,divide_distributions,n_bins,n_vertices_target):
        self.n_vertices_target= n_vertices_target
        self.divide_distributions = divide_distributions
        self.n_bins=n_bins
        self.df, *self.sample_normalization_parameters = process_dataset_for_knn(data_path,divide_distributions=self.divide_distributions)
        self.df_numeric = self.df.select_dtypes(include=np.number)
        self.faiss_knn = FaissKNeighbors(self.df_numeric,metric='L2')
        self.faiss_knn.train()
        self.custom_knn = CustomNeighbors(data_path)
        


        self.array_col =  ["bounding_box",
                    "angle_three_vertices","barycenter_vertice", "two_vertices",
                    "square_area_triangle", "cube_volume_tetrahedron" ]
    

    def query(self,model_path,n_samples_query,n_results,custom = False):
        vertices, element_dict, info = read_model(model_path)
        shape = Shape(vertices,element_dict,info)
        shape = process(shape,n_vertices_target=self.n_vertices_target)
        feature_dict = extract_features(shape,self.n_bins,n_samples=n_samples_query)
        feature_df = data_dict_parser(feature_dict)
        feature_df, _ = sample_normalizer(feature_df,*self.sample_normalization_parameters,divide_distributions=self.divide_distributions)
        feature_df_numeric = feature_df.select_dtypes(np.number)
        #Make sure columns identical and ordered
        assert list(feature_df_numeric.columns) == list(self.df_numeric.columns), "Column mismatch!"
        query_vector = feature_df_numeric.iloc[0,:].values.astype(np.float32)

        
     

        if not custom:

            distances, indices = self.faiss_knn.query(query_vector,n_results)
        else:
            distances, indices = self.custom_knn.query(query_vector,n_results)

        
        distances = distances.flatten().tolist() #Flatten batch dimension
        indices = indices.flatten().tolist()
        df_slice = self.df[self.df.index.isin(indices)]
        df_slice['distance'] = df_slice.index.map(lambda x:distances[indices.index(x)])
        
        
        #Add missing data to query df
        feature_df['file_name'] = str(model_path)
        feature_df['classification'] = 'query_input'
        feature_df['distance'] = 0
        # Put it at top of slice
        df_slice = pd.concat([df_slice,feature_df])
        df_slice = df_slice.sort_values('distance')

        #Send results for visualization

       
        return distances, indices, df_slice

        

  










if __name__ == '__main__':
    profiler = cProfile.Profile()
    
    data_path = Path("processed_data/data_coarse1_processed_10000_10000000.0.csv")
    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    plane_path = Path(r"data/benchmark/db/12/m1204/m1204.off")
    pig_path = Path(r"data/benchmark/db/1/m102/m102.off")
    watch_path = Path('data/benchmark/db/6/m601/m601.off')
    face_path = Path('data/benchmark/db/3/m302/m302.off')
    chess_piece_path = Path('data/benchmark/db/16/m1601/m1601.off')
    man_path = Path("data/benchmark/db/2/m201/m201.off")


    n_vertices_target = 10000
    query_interface = QueryInterface(data_path,divide_distributions=False,n_bins=10,n_vertices_target = n_vertices_target)
    
    path=pig_path
    profiler.run('query_interface.query(path,n_samples_query=1e+6,n_results=5)')
    profiler.dump_stats('query_profile_stats')
  



