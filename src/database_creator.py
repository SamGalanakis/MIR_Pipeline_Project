import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import bounding_box, cla_parser, calculate_diameter, align, angle_three_vertices, barycenter_vertice, two_vertices, cube_volume_tetrahedron, barycenter_vertice,square_area_triangle
from tqdm import tqdm
from shape import Shape
from file_reader import FileReader
import cProfile
import pyvista
import pyacvd
from preprocessing import process
from feature_extractor import extract_features

class Database:
    def __init__(self):
        self.classification_dict, self.hierarchy_dict, self.cla_info =  cla_parser(Path(r"data/benchmark/classification/v1/coarse1/coarse1Train.cla"))
        self.reader = FileReader()
        self.file_paths = []

    def create_database(self, database_name, apply_processing = True,n_faces_target=False):


        for root, dirs, files in os.walk(Path(r"data/benchmark")):
            for file in files:
                if file.endswith(".off"):
                    
                    self.file_paths.append(os.path.join(root, file))

        columns=["file_name","id","n_vertices","n_triangles","n_quads","bounding_box",
                "classification","volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
                "diameter","eccentricity", "angle_three_vertices","barycenter_vertice", "two_vertices",
                "square_area_triangle", "cube_volume_tetrahedron" ]


        data = {k:[] for k in columns}

        n_not_classified=0
        n_failed_subdivision=0
        n_failed_decimation=0
            
        for file in tqdm(self.file_paths):
            vertices, element_dict, info = self.reader.read(Path(file))
            shape = Shape(vertices,element_dict,info)


        
            
                
                

                
    
            if apply_processing:
                
                shape = process(shape,n_faces_target=n_faces_target)
             
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
            data["id"].append(id)




            #Get features
            feature_dict = extract_features(shape)

            #Add them to total data
            for key,val in feature_dict.items():
                data[key].append(val)
            
            
        n_classified_models = self.cla_info["n_models"]
        print(f"Missed/unclassified: {n_not_classified} of {len(self.file_paths)} of which {n_classified_models} are classified according to the cla.")
        


            
        path = (f"processed_data/{database_name}.csv")
        df = pd.DataFrame.from_dict(data,orient='columns')
        df.columns =columns
        df.to_csv(Path(path))
        print(f"Done making dataset and saved to {path}!")


if __name__=="__main__":
    database = Database()
    profiler= cProfile.Profile()
   # database.create_database("dataTest",apply_procesing=True,n_faces_target=1000)
    profiler.run('database.create_database("dataTest",apply_processing=True,n_faces_target=5000)')
    profiler.dump_stats("profiler_stats")
    




