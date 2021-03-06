from math import sin
import numpy as np
from pathlib import Path
from numpy.compat.py3k import asstr

import math
from file_reader import read_model
from scipy.spatial import ConvexHull, distance_matrix
import collections
import os



def get_princeton_classifications(cla_path_train):
    classification_dict_train, hierarchy_dict_train, cla_info_train =  cla_parser(Path(cla_path_train))
    test_path = cla_path_train.replace('train','test')
    test_path = test_path.replace('Train','Test')
    classification_dict_test, hierarchy_dict_test, cla_info_test = cla_parser(Path(test_path))
    classification_dict = merge_dicts(classification_dict_train,classification_dict_test)

    return classification_dict



def normalize_verts(verts):
    assert verts.shape[1]==3
    max_val = np.max(verts)
    min_val = np.min(verts)

    return (verts-min_val)/(max_val-min_val)
def get_all_file_paths(directory,extension):
    '''Get all file paths of given extension for files under given directory '''
    file_paths = []
    for root, dirs, files in os.walk(Path(directory)):
        for file in files:
            if file.endswith(extension):
                
                file_paths.append(os.path.join(root, file))
    return file_paths

def merge_dicts(dict1, dict2):
    res = {**dict1, **dict2}
    return res
        


def is_array_col(array_columns,col_name):
    #Matching array names to numbered columns or returning False
    for y in array_columns:
        if y in col_name and col_name.replace(y+'_',"").isdigit():
            return y
    return False

def vertice_sampler(vertices,n_samples,times,replace=True):
    #Make sure scientific notation is interpreted as int not float
    n_samples=int(n_samples)
    
    return [vertices[np.random.choice(vertices.shape[0],n_samples,replace=replace)] for x in range(times)]

def bounding_box(vertices,indices):
        as_points = vertices.reshape((-1, 3))

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



   
    return diam

def bin_count(input,n_bins):
    
    
    bins = np.linspace(np.min(input),np.max(input), 10)
    binned = np.digitize(input, bins)
    count_collections= collections.Counter(binned)
    counts = np.zeros(10)
    for key, val in count_collections.items():
        counts[key-1] = val
    return counts

def angle_three_random_vertices(vertices,n_samples,n_bins=10):
    vertices = vertices.reshape((-1,3))
    sample = vertice_sampler(vertices,n_samples,3)
    vecs1 = sample[0]-sample[1]
    norm_vecs1 = np.linalg.norm(vecs1,ord=2,axis=1,keepdims=True)
    vecs1= np.divide(vecs1,norm_vecs1,out=vecs1,where= norm_vecs1>0 )
    
    vecs2 = sample[0]-sample[2]
    norm_vecs2 = np.linalg.norm(vecs2,ord=2,axis=1,keepdims=True)
    vecs2= np.divide(vecs2,norm_vecs2,out=vecs2,where= norm_vecs2>0 )
    angles = np.arccos(vecs1*vecs2).sum(axis=1)
    
    counts = bin_count(angles,n_bins)
    return counts/sum(counts)


def barycenter_vertice(vertices, barycenter,n_samples,n_bins=10):
    vertices = vertices.reshape((-1,3))
    sample = vertice_sampler(vertices,n_samples,1)
    distances = np.linalg.norm(sample[0]-barycenter,axis=1)
    counts = bin_count(distances,n_bins)
    return counts/sum(counts)


def two_vertices(vertices,n_samples,n_bins=10):
    vertices = vertices.reshape((-1,3))
    sample = vertice_sampler(vertices,n_samples,2)

    distances = np.linalg.norm(sample[0]-sample[1],axis=1)

    counts = bin_count(distances,n_bins)
    return counts/sum(counts)

def square_area_triangle(vertices,n_samples,n_bins=10):
    vertices = vertices.reshape((-1,3))
    sample = vertice_sampler(vertices,n_samples,3)
    vecs1 = sample[0]-sample[1]
    vecs2 = sample[0]-sample[2]
    areas = np.sqrt(np.linalg.norm(np.cross(vecs1, vecs2),axis=1)/2)
    counts = bin_count(areas,n_bins)
    return counts/sum(counts)

def cube_volume_tetrahedron(vertices,n_samples,n_bins=10):
    vertices = vertices.reshape((-1,3))
    sample = vertice_sampler(vertices,n_samples,4)
    vecs1 = sample[1]-sample[0]
    vecs2 = sample[2]-sample[0]
    vecs3 = sample[3]-sample[0]
    areas = np.power(np.abs(np.linalg.det(np.stack([vecs1, vecs2,vecs3],axis=2)))/6,1/3)
    counts = bin_count(areas,n_bins)
    return counts/sum(counts)

def volume_original(vertices,triangles):
    vertices = vertices.reshape((-1,3))
    
    vertices_used = vertices[triangles].reshape(-1,3)
    
    # https://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up
    p1 =  vertices_used[0::3]
    p2 =  vertices_used[1::3]
    p3 =  vertices_used[2::3]
    
    return (p1*np.cross(p2,p3)).sum()/6

def volume(vertices,triangles):
    try:
        hull = ConvexHull(vertices)
    
    except:
        print("Could not calculate hull for diameter")
        return 0.1
    return hull.volume
    
if __name__ == "__main__":
  
   # coarse_1 = get_princeton_classifications(r"data\benchmark\classification\v1\coarse1\coarse1Train.cla")
    base = get_princeton_classifications(r"data\benchmark\classification\v1\base\train.cla")
    
    path  = Path(r"data/cube2.off")
    path = Path('data/benchmark/db/0/m0/m0.ply')
    vertices, element_dict, info = read_model(path)
    print(volume(vertices,element_dict["triangles"]))
    #angle_three_random_vertices(vertices,n_samples=1e+6)
    #barycenter_vertice(vertices,np.zeros(3),n_samples=1000)
    #two_vertices(vertices,n_samples=1000)
    #square_area_triangle(vertices,n_samples=1000)
    #cube_volume_tetrahedron(vertices,n_samples=1000)


