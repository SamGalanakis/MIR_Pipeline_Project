import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import bounding_box, cla_parser
from tqdm import tqdm
from shape import Shape
from file_reader import FileReader


classification_dict, hierarchy_dict, cla_info =  cla_parser(Path(r"data/benchmark/classification/v1/coarse1/coarse1Train.cla"))
reader = FileReader()
file_paths = []
for root, dirs, files in os.walk(Path(r"data/benchmark")):
    for file in files:
        if file.endswith(".off"):
             
             file_paths.append(os.path.join(root, file))

columns=["file_name","id","n_vertices","n_triangles","n_quads","bounding_box","barycenter","classification","volume","surface_area","surface_area"]


<<<<<<< HEAD
=======

>>>>>>> refs/remotes/origin/master
data = {k:[] for k in columns}

n_not_classified=0
for file in tqdm(file_paths):
    vertices, element_dict, info = reader.read(Path(file))
    shape = Shape(vertices,element_dict,info)
<<<<<<< HEAD
    
=======
    id = file.split("/")[-1].split(".")[-2].replace("m","")
>>>>>>> refs/remotes/origin/master

    id = file.split("/")[-1].split(".")[-2].replace("m","")
    try:
        classification = classification_dict[id]
        print(classification)
    except:
        n_not_classified +=1
        continue

    shape.make_pyvista_mesh()
    print(id)
    data["file_name"].append(file)
    data["id"].append(id)
    data["n_vertices"].append(shape.n_vertices)
    data["n_triangles"].append(shape.n_triangles)
    data["n_quads"].append(shape.n_quads)
    data["bounding_box"].append(str(shape.bounding_rect_vertices))
    data["barycenter"].append(str(shape.barycenter))
    data["classification"].append(classification)
    data["volume"].append(shape.pyvista_mesh.volume)
    data["surface_area"].append(sum(shape.pyvista_mesh.compute_cell_sizes(area = True, volume=False).cell_arrays["Area"]))
    xmax, ymax, zmax = shape.bounding_rect_vertices.reshape(-1 ,3).max(axis=0)
    if xmax != 1 or ymax != 1 or zmax != 1:
        print("asd")
    print()
n_classified_models = cla_info["n_models"]
print(f"Missed/unclassified: {n_not_classified} of {len(file_paths)} of which {n_classified_models} are classified according to the cla.")
    
    
df = pd.DataFrame.from_dict(data)
df.to_csv(Path(r"processed_data/data.csv"))
print("done")





