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

class Database:
    def __init__(self):
        self.classification_dict, self.hierarchy_dict, self.cla_info =  cla_parser(Path(r"data/benchmark/classification/v1/coarse1/coarse1Train.cla"))
        self.reader = FileReader()
        self.file_paths = []

    def create_database(self, database_name, apply_procesing = False,n_faces_target=False):

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

            shape = process(shape,n_faces_target=n_faces_target)

            id = os.path.basename(file).split(".")[0].replace("m","")
            if id in self.classification_dict.keys():
                classification = self.classification_dict[id]
                
            else:
                n_not_classified +=1
                classification = None
                

            
            print(id)
            data["file_name"].append(file)
            data["id"].append(id)
            data["n_vertices"].append(shape.n_vertices)
            data["n_triangles"].append(shape.n_triangles)
            data["n_quads"].append(shape.n_quads)
            data["bounding_box"].append(str(shape.bounding_rect_vertices))
      
            data["classification"].append(classification)
            data["volume"].append(np.maximum(shape.pyvista_mesh.volume,0.01))#clamp to avoid 0 volume for 2d models
            
            data["surface_area"].append(shape.pyvista_mesh.area)
            bounding_box_sides = shape.bounding_rect_vertices.reshape((-1 ,3)).max(axis=0)-shape.bounding_rect_vertices.reshape((-1 ,3)).min(axis=0)
            bounding_box_sides = np.maximum(bounding_box_sides,0.01) #clamp above so no zero division for essentially 2d models
            data["bounding_box_ratio"].append(np.max(bounding_box_sides)/np.min(bounding_box_sides))
            data["compactness"].append(np.power(data["surface_area"][-1],3) / np.power(data["volume"][-1],2))
            data["bounding_box_volume"].append(np.prod(bounding_box_sides)) 
            data["diameter"].append(calculate_diameter(shape.vertices))
            data["eccentricity"].append(np.max(shape.eigenvalues)/np.maximum(np.min(shape.eigenvalues),0.01)) #also clamp
            #Histograms
            data["angle_three_vertices"].append(angle_three_vertices(shape.vertices))
            data["barycenter_vertice"].append(barycenter_vertice(shape.vertices, shape.barycenter))
            data["two_vertices"].append(two_vertices(shape.vertices))
            data["square_area_triangle"].append(square_area_triangle(shape.vertices))
            data["cube_volume_tetrahedron"].append(cube_volume_tetrahedron(shape.vertices))


            
            
        n_classified_models = self.cla_info["n_models"]
        print(f"Missed/unclassified: {n_not_classified} of {len(self.file_paths)} of which {n_classified_models} are classified according to the cla.")
        
        if n_faces_target:
            print(f"Failed subdivision on {n_failed_subdivision}, failed decimation on {n_failed_decimation}")

            
        path = (f"processed_data/{database_name}.csv")
        df = pd.DataFrame.from_dict(data,orient='columns')
        df.columns =columns
        df.to_csv(Path(path))
        print("done")


if __name__=="__main__":
    database = Database()
    profiler= cProfile.Profile()
   # database.create_database("dataTest",apply_procesing=True,n_faces_target=1000)
    profiler.run('database.create_database("dataTest",apply_procesing=True,n_faces_target=1000)')
    profiler.dump_stats("profiler_stats")
    




