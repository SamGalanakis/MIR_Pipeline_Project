import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import bounding_box, cla_parser
from tqdm import tqdm
from shape import Shape
from file_reader import FileReader


classification_dict, hierarchy_dict, cla_info =  cla_parser(Path(r"data\benchmark\classification\v1\coarse1\coarse1Train.cla"))
reader = FileReader()
file_paths = []
for root, dirs, files in os.walk(Path(r"data\benchmark")):
    for file in files:
        if file.endswith(".off"):
             
             file_paths.append(os.path.join(root, file))

columns=["id","n_vertices","n_triangles","n_quads","bounding_box","barycenter","classification"]



id_list =[]
n_vertices_list = []
n_triangles_list =[]
n_quads_list=[]
bounding_box_list =[]
barycenter_list = []
data = {k:[] for k in columns}

n_not_classified=0
for file in tqdm(file_paths[0:10]):
    vertices, element_dict, info = reader.read(Path(file))
    shape = Shape(vertices,element_dict,info)
    id = file.split("\\")[-1].split(".")[-2].replace("m","")

    try:
        classification = classification_dict[id]
        print(classification)
    except:
        n_not_classified +=1
        continue
    print(id)
    data["id"].append(id)
    data["n_vertices"].append(shape.n_vertices)
    data["n_triangles"].append(shape.n_triangles)
    data["n_quads"].append(shape.n_quads)
    data["bounding_box"].append(str(shape.bounding_rect_vertices))
    data["barycenter"].append(str(shape.barycenter))
    data["classification"].append(classification)
    
n_classified_models = cla_info["n_models"]
print(f"Missed/unclassified: {n_not_classified} of {len(file_paths)} of which {n_classified_models} are classified according to the cla.")
    
    
df = pd.DataFrame.from_dict(data)
df.to_csv(Path(r"processed_data/data.csv"))
print("done")





