import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import cla_parser, merge_dicts, get_all_file_paths, get_princeton_classifications
from tqdm import tqdm
from shape import Shape
from file_reader import read_model
import cProfile
from preprocessing import process
from feature_extractor import extract_features
import concurrent
import math
import warnings 
import concurrent.futures
def data_dict_parser(data_dict):
        if  isinstance(list(data_dict.values())[0],list):
            lengths  = {key:(val[0].size if isinstance(val[0],np.ndarray) else 1 ) for key,val  in data_dict.items() }
            col_length = len(list(data_dict.values())[0])
            
            
        else:
            lengths  = {key:(val.size if isinstance(val,np.ndarray) else 1 ) for key,val  in data_dict.items() }
            col_length=1
            
        df = pd.DataFrame(index = range(col_length)) 
        row_length = sum(lengths.values())
            
        
        for key in sorted(data_dict):
            val = data_dict[key]
            length = lengths[key]
            if length ==1:
                col_names = key
                
            else:
                col_names =  [f"{key}_{num}" for num in range(length)]
                val = np.array(val,dtype=np.float32)
            
            df[col_names] = val
        assert row_length == df.shape[1] , "Shape mismatch!"
        return df

class Database:
    def __init__(self):
        
        cla_path = 'data/benchmark/classification/v1/base/train.cla'
        cla_path = 'data/benchmark/classification/v1/coarse1/coarse1Train.cla'
        
        self.classification_dict = get_princeton_classifications(cla_path)
        unique_classes = len(set(self.classification_dict.values()))
        print(f'{unique_classes} unique classes')
        
  


        self.columns=["file_name","n_vertices","n_triangles","n_quads",
                "classification","volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
                "diameter","eccentricity" ]
        self.col_array = ["bounding_box","angle_three_vertices","barycenter_vertice", "two_vertices",
                "square_area_triangle", "cube_volume_tetrahedron"]





    def process_subset(self,file_list,apply_processing,n_vertices_target,n_bins,process_index):
        print(f' {process_index} : Starting subset processor!')
        data_subset = {k:[] for k in self.columns+self.col_array}
        for index, file in enumerate(file_list):

            if index % 50 ==0: 
                print(f' {process_index} : Is at {index}/{len(file_list)}!')


            vertices, element_dict, info = read_model(Path(file))
            shape = Shape(vertices,element_dict,info)


            if apply_processing:
                
                shape = process(shape,n_vertices_target=n_vertices_target)
            
            else:
                shape.make_pyvista_mesh()

            id = os.path.basename(file).split(".")[0].replace("m","")
            if id in self.classification_dict.keys():
                classification = self.classification_dict[id]
                
            else:
            
                classification = None
                

            
            data_subset["classification"].append(classification)
            data_subset["file_name"].append(file)





            #Get features
            feature_dict = extract_features(shape,n_bins=n_bins,n_samples=self.n_samples)

            #Add them to total data

            
            for key,val in feature_dict.items():
                data_subset[key].append(val)
        print(f'{process_index} : Finished!')
        return data_subset

    def create_database(self, database_name,n_samples,n_bins, apply_processing = True,n_vertices_target=False,n_processes=4):
        self.n_samples = n_samples
        self.file_paths = get_all_file_paths(r'data/benchmark','.off')

     

        
        
        data = {k:[] for k in self.columns+self.col_array}
        subset_lengh  = math.ceil(len(self.file_paths)/n_processes)
        input_subsets = [self.file_paths[x*subset_lengh:(x+1)*subset_lengh] for x in range(n_processes)  ]
        assert sum([len(x) for x in input_subsets]) == len(self.file_paths)
        f = lambda x: self.process_subset(x,apply_processing,n_vertices_target,n_bins)
        with concurrent.futures.ProcessPoolExecutor() as executor:
           
            result_subsets = [executor.submit(self.process_subset,*(file_list,apply_processing,n_vertices_target,n_bins,process_index))
             for process_index,file_list in enumerate(input_subsets)]
            print('Starting!')
            for result_subset in concurrent.futures.as_completed(result_subsets):
                for key,val in result_subset.result().items():
                    data[key].extend(val)

        df = data_dict_parser(data) 

        processed = "processed" if apply_processing else ""
        database_name = f"{database_name}_{processed}_{n_vertices_target}_{n_samples}"
        
        path = f"processed_data/{database_name}.csv"
        
        
        df.to_csv(Path(path))
        print(f"Done making dataset and saved to {path}!")


if __name__=="__main__":
    warnings.filterwarnings('ignore')
    database = Database()
    profiler= cProfile.Profile()
    base_name = 'data_coarse1'
    n_samples = 10e+6
    apply_processing = True
    n_vertices_target = 10000
    n_bins=10
    
    profiler.run('database.create_database(base_name,n_samples=n_samples,apply_processing=apply_processing,n_vertices_target=n_vertices_target,n_bins=10)')
    profiler.dump_stats("profiler_stats")

    




