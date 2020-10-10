import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import cla_parser, merge_dicts
from tqdm import tqdm
from shape import Shape
from file_reader import FileReader
import cProfile
from preprocessing import process
from feature_extractor import extract_features



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
        n_vertices_target
        cla_path = 'data/benchmark/classification/v1/base/train.cla'
        classification_dict_train, self.hierarchy_dict_train, self.cla_info_train =  cla_parser(Path(cla_path))
        classification_dict_test, self.hierarchy_dict_test, self.cla_info_test = cla_parser(Path(cla_path.replace('train','test')))
        self.classification_dict = merge_dicts(classification_dict_train,classification_dict_test)
        unique_classes = len(set(self.classification_dict.values()))
        print(f'{unique_classes} unique classes')
        self.reader = FileReader()
        self.file_paths = []

    def create_database(self, database_name,n_samples,n_bins=10, apply_processing = True,n_vertices_target=False):


        for root, dirs, files in os.walk(Path(r"data/benchmark")):
            for file in files:
                if file.endswith(".off"):
                    
                    self.file_paths.append(os.path.join(root, file))

        columns=["file_name","n_vertices","n_triangles","n_quads",
                "classification","volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
                "diameter","eccentricity" ]
        col_array = ["bounding_box","angle_three_vertices","barycenter_vertice", "two_vertices",
                "square_area_triangle", "cube_volume_tetrahedron"]

        
        
        data = {k:[] for k in columns+col_array}

    
        
            
        for file in tqdm(self.file_paths):
            vertices, element_dict, info = self.reader.read(Path(file))
            shape = Shape(vertices,element_dict,info)

    
            if apply_processing:
                
                shape = process(shape,n_vertices_target=n_vertices_target)
             
            else:
                shape.make_pyvista_mesh()

            id = os.path.basename(file).split(".")[0].replace("m","")
            if id in self.classification_dict.keys():
                classification = self.classification_dict[id]
                
            else:
                n_not_classified +=1
                classification = None
                

            
            data["classification"].append(classification)
            data["file_name"].append(file)
       




            #Get features
            feature_dict = extract_features(shape,n_bins=n_bins,n_samples=n_samples)

            #Add them to total data

            
            for key,val in feature_dict.items():
                data[key].append(val)
            
            
        #new
      

        df = data_dict_parser(data) 




        #new
        # n_classified_models = self.cla_info["n_models"]
        # print(f"Missed/unclassified: {n_not_classified} of {len(self.file_paths)} of which {n_classified_models} are classified according to the cla.")
        
        
        
       
        processed = "processed" if apply_processing else ""
        database_name = f"{database_name}_{processed}_{n_vertices_target}_{n_samples}"
        
        path = f"processed_data/{database_name}.csv"
        
        
        df.to_csv(Path(path))
        print(f"Done making dataset and saved to {path}!")


if __name__=="__main__":
    database = Database()
    profiler= cProfile.Profile()
    base_name = 'data'
    n_samples = 1e+6
    apply_processing = True
    n_vertices_target = 10000
    profiler.run('database.create_database(base_name,n_samples=n_samples,apply_processing=apply_processing,n_vertices_target=n_vertices_target)')
    profiler.dump_stats("profiler_stats")

    




