import numpy as np
import pyvista
from model_viewer import ModelViewer
from pathlib import Path
from file_reader import read_model
from utils import bounding_box, align, flip_test, normalize_verts
from itertools import combinations
import pyvista as pv
import math
from pyacvd import Clustering

class Shape:
    def __init__(self,vertices, element_dict, info):
        self.vertices=vertices.reshape((-1,3))
        self.element_dict = element_dict
        self.info = info
        self.n_triangles = element_dict["triangles"].size/3
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size/3
        self.viewer = ModelViewer()
        
        self.pyvista_mesh = False
    
        
        

    def normalize(self):
        
        self.vertices = normalize_verts(self.vertices)
        return self.vertices
    def barycenter_to_origin(self):
        barycenter = self.vertices.mean(axis=0)
        self.vertices -= barycenter

    def align(self):
        self.vertices, self.eigenvectors, self.eigenvalues = align(self.vertices.flatten())
    
    def flip(self):
        self.vertices = flip_test(self.vertices,self.element_dict["triangles"]).astype(np.float32)
        
    def bounding_rect(self):
        self.bounding_rect_vertices, self.bounding_rect_indices = bounding_box(self.vertices.reshape(-1, 3) ,self.element_dict["triangles"])


    def process_shape(self):

     
        self.normalize()

        self.align()
        
        self.flip()

        self.barycenter_to_origin()

        self.bounding_rect()
    

  
    

    
    def view(self):
        self.viewer.process(vertices = self.vertices.flatten() , indices = self.element_dict["triangles"],info=self.info)

    def pyvista_mesh_to_base(self,pyvista_mesh):
        self.element_dict["triangles"] = pyvista_mesh.faces.reshape((-1,4))[:,1:].astype(np.uint32)
        self.vertices = np.array(pyvista_mesh.points.reshape((-1,3))).astype(np.float32)

        self.n_vertices=self.vertices.size/3
        self.n_triangles = self.element_dict["triangles"].size/3
        
        

    def make_pyvista_mesh(self):
        triangles = np.zeros((self.element_dict["triangles"].shape[0],4)) +3
        triangles [:,1:4] = self.element_dict["triangles"]
        triangles = np.array(triangles,dtype=np.int)
        self.pyvista_mesh = pv.PolyData(self.vertices,triangles)


        

if __name__ == "__main__":
    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    path = Path(r"data/test.ply")
    
    max_path = Path('data/benchmark/db/17/m1755/m1755.off')
    problem_path = "data/benchmark/db/2/m201/m201.off"
    pig_path = Path(r"data\benchmark\db\1\m102\m102.off")
    path = path
    
  
    vertices, element_dict, info = read_model(path)
    shape = Shape(vertices,element_dict,info)

    shape.view()
    shape.make_pyvista_mesh()
    
    shape.process_shape()

  

    
    
    










