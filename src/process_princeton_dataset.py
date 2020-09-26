import numpy as np
import pandas as pd
import os
from pathlib import Path
from utils import bounding_box, cla_parser, calculate_diameter, align, angle_three_vertices, barycenter_vertice, two_vertices, cube_volume_tetrahedron, barycenter_vertice,square_area_triangle
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

columns=["file_name","id","n_vertices","n_triangles","n_quads","bounding_box","barycenter",
        "classification","volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
        "diameter","eccentricity", "angle_three_vertices","barycenter_vertice", "two_vertices",
        "square_area_triangle", "cube_volume_tetrahedron" ]


data = {k:[] for k in columns}

n_not_classified=0
for file in tqdm(file_paths):
    vertices, element_dict, info = reader.read(Path(file))
    shape = Shape(vertices,element_dict,info)
    

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
    
    axis_size = shape.bounding_rect_vertices.reshape(-1 ,3).max(axis=0)
    data["bounding_box_ratio"].append(np.max(axis_size)/np.min(axis_size))
    
    data["compactness"].append(np.power(data["surface_area"][-1],3) / np.sqrt(data["volume"][-1]))
    data["bounding_box_volume"].append(np.prod(axis_size))
    data["diameter"].append(calculate_diameter(shape.vertices))
    
    #TODO
    data["eccentricity"].append(shape.eigenvectors)
    #Histograms
    a = angle_three_vertices(shape.vertices)
    data["angle_three_vertices"].append(angle_three_vertices(shape.vertices))
    data["barycenter_vertice"].append(barycenter_vertice(shape.vertices, shape.barycenter))
    data["two_vertices"].append(two_vertices(shape.vertices))
    data["square_area_triangle"].append(square_area_triangle(shape.vertices))
    data["cube_volume_tetrahedron"].append(cube_volume_tetrahedron(shape.vertices))




    print()
    
n_classified_models = cla_info["n_models"]
print(f"Missed/unclassified: {n_not_classified} of {len(file_paths)} of which {n_classified_models} are classified according to the cla.")
    
    
df = pd.DataFrame.from_dict(data)
df.to_csv(Path(r"processed_data/data.csv"))
print("done")





