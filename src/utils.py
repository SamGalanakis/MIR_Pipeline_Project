import numpy as np
from pathlib import Path
import pyrr
import math
from file_reader import FileReader
from scipy.spatial import ConvexHull



def bounding_box(vertices,indices):
        as_points = vertices.reshape(-1, 3)



        max_x, max_y, max_z = as_points.max(axis=0)
        min_x, min_y, min_z = as_points.min(axis=0)

        bounding_rect_vertices = np.array([[min_x,min_y,min_z],[max_x,min_y,min_z],[max_x,min_y,max_z],[min_x,min_y,max_z],
                            [min_x,max_y,min_z],[max_x,max_y,min_z],[max_x,max_y,max_z],[min_x,max_y,max_z]]).flatten()
        
        bounding_rect_indices = (np.array([0,1,2,2,3,0,  4,5,6,6,7,4, 0,1,4,4,5,1,  2,3,6,6,7,3, 0,3,4,4,7,3, 1,2,5,5,6,2 ],dtype=np.uint32) + indices.max()+1).flatten()

        return bounding_rect_vertices, bounding_rect_indices




def cla_parser(path):
    classification_dict = {}
    hierarchy_dict = {}
    assert(path.suffix == ".cla", "Not a cla file!")
    with open(path) as f:
        lines_list = f.readlines()
    lines_list = [x.strip() for x in lines_list]
    lines_list = [x for x in lines_list if len(x)>0]

    psb_version = lines_list[0].split()[1]
    n_classes, n_models = lines_list[1].split()
    info = {"n_classes":n_classes, "n_models":n_models}
    print(f"Classes: {n_classes}, Models: {n_models}")
    lines_list = lines_list[2:]

    current_class = ""
    
    for line in lines_list:
        if line.isdigit():
            classification_dict[line] = current_class
        else:
            current_class , parent_class , n_class_models = line.split()
            if parent_class != "0":
                hierarchy_dict[current_class]=parent_class
            

    return classification_dict, hierarchy_dict, info



def align(vertices):
    vertices= vertices.reshape((-1,3))
    cov = np.cov(vertices,rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    eig_indices = sorted(range(0,3),key = lambda x : eigenvalues[x],reverse=True)
    
    eigenvecs_sorted = np.zeros(eigenvectors.shape)
    for count,  x in enumerate(eig_indices):
        eigenvecs_sorted[:,count] = eigenvectors[:,x]
 
    vertices = np.matmul(vertices,eigenvectors)
    
    return  np.array(vertices.flatten(),dtype=np.float32)

def calculate_diameter(vertices):
    hull = ConvexHull(vertices)
    diameter = 0
    for idx, simplice in enumerate(hull.simplices):
        if idx + 1 >= len(hull.simplices):
            diameter += (np.linalg.norm(simplice - hull.simplices[0]))
        else:
            diameter += (np.linalg.norm(simplice - hull.simplices[idx+1]))

    return diameter

    
if __name__ == "__main__":
    
    #cla_parser(Path(r"data\benchmark\classification\v1\coarse1\coarse1Train.cla"))
    reader= FileReader()
    path = path = Path(r"data/test.ply")
    vertices, element_dict, info = reader.read(path)
    align(vertices)
    