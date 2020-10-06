
from shape import Shape
from file_reader import FileReader
from utils import  align, angle_three_random_vertices, calculate_diameter, barycenter_vertice, two_vertices, cube_volume_tetrahedron, barycenter_vertice,square_area_triangle
from pathlib import Path
import numpy as np


def extract_features(shape):
    '''Input must be Shape or path'''
    if isinstance(shape,Shape):
        pass
    elif isinstance(shape,Path) or isinstance(shape,str):
        shape = Path(shape)
        reader = FileReader()
        shape = Shape(*reader.read(shape))
    else:
        raise Exception("Input must be Shape or path")

    
    feature_dict = {}

    
    feature_dict["n_vertices"]=shape.n_vertices
    feature_dict["n_triangles"]=shape.n_triangles
    feature_dict["n_quads"]=shape.n_quads
    feature_dict["bounding_box"]=shape.bounding_rect_vertices

    feature_dict["volume"]=np.maximum(shape.pyvista_mesh.volume,0.01)#clamp to avoid 0 volume for 2d models

    feature_dict["surface_area"]=shape.pyvista_mesh.area
    bounding_box_sides = shape.bounding_rect_vertices.reshape((-1 ,3)).max(axis=0)-shape.bounding_rect_vertices.reshape((-1 ,3)).min(axis=0)
    bounding_box_sides = np.maximum(bounding_box_sides,0.01) #clamp above so no zero division for essentially 2d models
    feature_dict["bounding_box_ratio"]=np.max(bounding_box_sides)/np.min(bounding_box_sides)
    feature_dict["compactness"]=np.power(feature_dict["surface_area"],3) / (36 * np.pi * np.power(feature_dict["volume"],2))
    feature_dict["bounding_box_volume"]=np.prod(bounding_box_sides)
    feature_dict["diameter"]=calculate_diameter(shape.vertices)
    
    *_, eigenvalues = align(shape.vertices)

    feature_dict["eccentricity"]=np.max(eigenvalues)/np.maximum(np.min(eigenvalues),0.01) #also clamp
    #Histograms
    feature_dict["angle_three_vertices"]  = angle_three_random_vertices(shape.vertices)
    feature_dict["barycenter_vertice"]=barycenter_vertice(shape.vertices, np.zeros(3,dtype=np.float32))
    feature_dict["two_vertices"]=two_vertices(shape.vertices)
    feature_dict["square_area_triangle"]=square_area_triangle(shape.vertices)
    feature_dict["cube_volume_tetrahedron"]=cube_volume_tetrahedron(shape.vertices)


    return feature_dict













    return feature_dict

if __name__=='__main__':
    path = Path(r"data/test.ply")
    path = Path(r"data/benchmark/db/1/m102/m102.off")
    extract_features(path)