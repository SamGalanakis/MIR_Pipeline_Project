from math import sin
import numpy as np
from pathlib import Path
from numpy.compat.py3k import asstr
import pyrr
import math
from file_reader import FileReader
from scipy.spatial import ConvexHull, distance_matrix
from scipy.stats import binned_statistic
import collections
import time
import itertools
import pandas as pd

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
    assert path.suffix == ".cla", "Not a cla file!"
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
   
    vertices = vertices.reshape((3,-1),order="F")
    cov = np.cov(vertices)
    eigenvalues, eigenvectors = np.linalg.eig(cov)

    eig_indices = sorted(range(0,3),key = lambda x : eigenvalues[x],reverse=True)
    
    eigenvecs_sorted = np.zeros(eigenvectors.shape)
    for count,  x in enumerate(eig_indices):
        eigenvecs_sorted[:,count] = eigenvectors[:,x]

    transformation = np.linalg.inv(eigenvecs_sorted)

    vertices = np.matmul(transformation,vertices)

    
    return  vertices.flatten(order="F").reshape((-1,3)).astype(np.float32) ,  eigenvectors, eigenvalues



def flip_test(vertices,triangle_indices):
    
    f = lambda x: vertices[x,:]
    triangles = f(triangle_indices)

    centroids = np.sum(triangles,axis=1)/3

    second_moments = np.multiply( np.sign(centroids) , centroids**2).sum(axis=0)

    transformation = np.zeros((3,3))
 
    np.fill_diagonal(transformation,np.sign(second_moments) )


    

    return np.matmul(vertices,transformation)
   

    
    
 




def calculate_diameter(vertices):
    
    vertices=vertices.reshape((-1,3))
    try:
        hull = ConvexHull(vertices)
        
    except:
        print("Could not calculate hull for diameter")
        return None



    f = lambda x : vertices[x,:]
    unique_hull_points = f(np.unique(hull.simplices))

    diam = distance_matrix(unique_hull_points,unique_hull_points).max()


    
   # Naive algorithm, quite inefficient can take ~ 3s for larger model
   # diam = itertools.combinations(list(unique_hull_points),2).max()
  #  diam = max([np.linalg.norm(x[0]-x[1]) for x in point_pairs])
   
    return diam
        

def calculate_angle(a, b, c):
    """
        Calculates angle between three vertices
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return angle

def angle_three_vertices(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.floor(len(vertices) * 0.8)
    while number_of_vertices % 3 != 0:
        number_of_vertices += 1

    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    angles_three_vertices = [calculate_angle(a, b, c) for a, b, c in grouped(vertices[indices], 3)]
    
    bins = np.linspace(np.min(angles_three_vertices),np.max(angles_three_vertices), 10)
    binned = np.digitize(angles_three_vertices, bins)
    count_collections= collections.Counter(binned)
    counts = np.zeros(10)
    for key, val in count_collections.items():
        counts[key-1] = val
    
    return counts , angle_three_vertices


def barycenter_vertice(vertices, barycenter):
    vertices = vertices.reshape(-1,3)
    indices = np.random.choice(vertices.shape[0], int(len(vertices) * 0.8), replace=False)

    barycenter_vertices = [np.linalg.norm(vertice - barycenter) for vertice in vertices[indices]]

    bins = np.linspace(np.min(barycenter_vertices),np.max(barycenter_vertices), 10)
    binned = np.digitize(barycenter_vertices, bins)
    count_collections= collections.Counter(binned)
    counts = np.zeros(10)
    for key, val in count_collections.items():
        counts[key-1] = val
    return counts , barycenter_vertice




def two_vertices(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    if number_of_vertices % 2 != 0:
        number_of_vertices += 1
    
    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    
    vertices_difference = [np.linalg.norm(a - b) for a, b in grouped(vertices, 2)]

    bins = np.linspace(np.min(vertices_difference),np.max(vertices_difference), 10)
    binned = np.digitize(vertices_difference, bins)
    count_collections= collections.Counter(binned)
    counts = np.zeros(10)
    for key, val in count_collections.items():
        counts[key-1] = val
  
        
    return counts , vertices_difference

def square_area_triangle(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    while number_of_vertices % 3 != 0:
        number_of_vertices += 1

    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    areas = [1/2 * np.linalg.norm((a[0]-c[0])*(b[1]-a[1]) - (a[0] - b[0])*(c[1]-a[1])) for a,b ,c in grouped(vertices[indices], 3)]

    bins = np.linspace(np.min(areas),np.max(areas), 10)
    binned = np.digitize(areas, bins)
    counts = np.zeros(10)
    for bin_ in binned:
        counts[bin_ -1] += 1
        
    return counts , areas

def cube_volume_tetrahedron(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    while number_of_vertices % 4 != 0:
        number_of_vertices += 1

    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    volumes = [np.cbrt(np.linalg.norm(np.dot(a-d, np.cross(b-d,c-d))) / 6) for a, b ,c, d in grouped(vertices[indices], 4)]

    bins = np.linspace(np.min(volumes),np.max(volumes), 10)
    binned = np.digitize(volumes, bins)
    count_collections= collections.Counter(binned)
    counts = np.zeros(10)
    for key, val in count_collections.items():
        counts[key-1] = val

    return counts , volumes


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

def parse_array_from_str(list_str):
    if type(list_str)!=str:
        print(f"{list_str} could not be parsed to an array, returning None")
        return None
    temp = np.array(list(map(lambda x: float(x),   list_str.replace("[","").replace("]","").split())))
    return temp

def model_feature_dist(comparison_model,df2,single_columns,array_columns,norm):
    
    single_dif = (comparison_model[single_columns].values - df2[single_columns].values).astype(np.float64)

    if type(comparison_model)==pd.Series and type(df2)==pd.Series:
        array_difs = (df2[array_columns] - comparison_model[array_columns] ).apply(lambda x: x.mean())
        temp = np.linalg.norm(np.append(single_dif,array_difs.values,0),ord=2,axis=0)
        return temp
    else:
        array_difs = (df2[array_columns] - comparison_model[array_columns] ).applymap(lambda x: x.mean())
        temp = np.linalg.norm(np.append(single_dif,array_difs.values,1),ord=2,axis=1)
        return temp
    
if __name__ == "__main__":
  
    #cla_parser(Path(r"data\benchmark\classification\v1\coarse1\coarse1Train.cla"))
    reader= FileReader()
    path = path = Path(r"data/test.ply")
    vertices, element_dict, info = reader.read(path)
    align(vertices)


