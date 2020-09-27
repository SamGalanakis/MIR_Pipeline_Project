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
    hull = ConvexHull(vertices.reshape(-1,3))
    diameter = 0
    for idx, simplice in enumerate(hull.simplices):
        if idx + 1 >= len(hull.simplices):
            diameter += np.linalg.norm(simplice - hull.simplices[0])
        else:
            diameter += np.linalg.norm(simplice - hull.simplices[idx+1])

    return diameter

def calculate_angle(a, b, c):
    """
        Calculates angle between three vertices
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)

    return np.degrees(angle)

def angle_three_vertices(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    while number_of_vertices % 3 != 0:
        number_of_vertices += 1

    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    angles_three_vertices = [calculate_angle(a, b, c) for a, b, c in grouped(vertices[indices], 3)]
    bins = np.linspace(0, 1, 10)
    binned = np.digitize(angles_three_vertices, bins)
    _ , counts = np.unique(binned, return_counts=True)

    return counts


def barycenter_vertice(vertices, barycenter):
    vertices = vertices.reshape(-1,3)
    indices = np.random.choice(vertices.shape[0], int(len(vertices) * 0.8), replace=False)

    barycenter_vertices = [np.linalg.norm(vertice - barycenter) for vertice in vertices[indices]]
    bins = np.linspace(0, 1, 10)
    binned = np.digitize(barycenter_vertices, bins)
    _ , counts = np.unique(binned, return_counts=True)

    return counts

def two_vertices(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    if number_of_vertices % 2 != 0:
        number_of_vertices += 1
    
    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    
    vertices_difference = [np.linalg.norm(a - b) for a, b in grouped(vertices, 2)]
    bins = np.linspace(0, 1, 10)
    binned = np.digitize(vertices_difference, bins)
    _, counts = np.unique(binned, return_counts=True)

    return counts

def square_area_triangle(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    while number_of_vertices % 3 != 0:
        number_of_vertices += 1

    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    areas = [1/2 * np.linalg.norm((a[0]-c[0])*(b[1]-a[1]) - (a[0] - b[0])*(c[1]-a[1])) for a,b ,c in grouped(vertices[indices], 3)]
    bins = np.linspace(0, 1, 10)
    binned = np.digitize(areas, bins)
    _ , counts = np.unique(binned, return_counts=True)

    return counts

def cube_volume_tetrahedron(vertices):
    vertices = vertices.reshape(-1,3)
    number_of_vertices = np.ceil(len(vertices) * 0.8)
    while number_of_vertices % 4 != 0:
        number_of_vertices += 1

    indices = np.random.choice(vertices.shape[0], int(number_of_vertices), replace=False)

    volumes = [np.cbrt(np.linalg.norm(np.dot(a-d, np.cross(b-d,c-d))) / 6) for a, b ,c, d in grouped(vertices[indices], 4)]
    bins = np.linspace(0, 1, 10)
    binned = np.digitize(volumes, bins)
    _ , counts = np.unique(binned, return_counts=True)

    return counts


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)

    
if __name__ == "__main__":
    
    #cla_parser(Path(r"data\benchmark\classification\v1\coarse1\coarse1Train.cla"))
    reader= FileReader()
    path = path = Path(r"data/test.ply")
    vertices, element_dict, info = reader.read(path)
    align(vertices)


