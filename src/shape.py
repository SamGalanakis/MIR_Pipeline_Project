import numpy as np
import pyvista
from model_viewer import ModelViewer
from pathlib import Path
from file_reader import FileReader
from utils import bounding_box, align, flip_test
from itertools import combinations
import pyvista as pv
import math
from pyacvd import Clustering

class Shape:
    def __init__(self,vertices, element_dict, info):
        self.vertices=vertices
        self.element_dict = element_dict
        self.info = info
        self.n_triangles = element_dict["triangles"].size/3
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size/3
        self.viewer = ModelViewer()
        
        self.pyvista_mesh = False
        self.barycenter = vertices.reshape(-1, 3).mean(axis=0) 
        self.normalized_barycentered_vertices = vertices.reshape(-1, 3).mean(axis=0) - self.barycenter
        self.bounding_rect_vertices, self.bounding_rect_indices = bounding_box(self.vertices.reshape(-1, 3) ,self.element_dict["triangles"])
        _, self.eigenvectors, self.eigenvalues = align(vertices)

    def get_edges(self):
        self.edges = set() 
        edges_non_unique = list(dict.fromkeys([item for t in [list(combinations(triangle,2)) for triangle in self.element_dict["triangles"]] for item in t]))

        for (a, b) in edges_non_unique:
            if (a, b) and (b ,a) not in self.edges:
                self.edges.add((a,b))


    def process_shape(self):

        max_range = 1
        min_range = 0
        processed_vertices = self.vertices.reshape((-1, 3)) 
        scaled_unit = (max_range - min_range) / (np.max(processed_vertices) - np.min(processed_vertices))

        self.processed_vertices = processed_vertices*scaled_unit - np.min(processed_vertices)*scaled_unit + min_range
        self.processed_vertices, self.eigenvectors, self.eigenvalues = align(self.processed_vertices.flatten())
        self.processed_vertices = flip_test(self.processed_vertices,self.element_dict["triangles"]).astype(np.float32)

        self.barycenter =   self.processed_vertices.reshape(-1, 3).mean(axis=0)
        self.processed_vertices = self.processed_vertices.reshape(-1, 3)  - self.barycenter
        self.bounding_rect_vertices, self.bounding_rect_indices = bounding_box(self.processed_vertices,self.element_dict["triangles"])
    

  
    

    def view_processed(self):
        self.viewer.process(vertices = self.processed_vertices,indices = self.element_dict["triangles"],info=self.info)
    def view(self):
        self.viewer.process(vertices = self.vertices , indices = self.element_dict["triangles"],info=self.info)

    def pyvista_mesh_to_base(self,pyvista_mesh):
        self.element_dict["triangles"] = pyvista_mesh.faces.reshape((-1,4))[:,1:].astype(np.uint32)
        self.vertices = np.array(pyvista_mesh.points.flatten())

        self.n_vertices=self.vertices.size/3
        self.n_triangles = self.element_dict["triangles"].size/3
        
        

    def make_pyvista_mesh(self):
        triangles = np.zeros((self.element_dict["triangles"].shape[0],4)) +3
        triangles [:,1:4] = self.element_dict["triangles"]
        triangles = np.array(triangles,dtype=np.int)
        self.pyvista_mesh = pv.PolyData(self.vertices.reshape(-1,3),triangles)
    def make_pyvista_mesh_processed(self):
        triangles = np.zeros((self.element_dict["triangles"].shape[0],4)) +3
        triangles [:,1:4] = self.element_dict["triangles"]
        triangles = np.array(triangles,dtype=np.int)
        self.pyvista_mesh_processed = pv.PolyData(self.processed_vertices.reshape(-1,3),triangles)
    


    def subdivide(self,times=1,algo="linear",target=False,undercut=True):
       
        
        if type(self.pyvista_mesh) == bool:
            self.make_pyvista_mesh()
        if undercut:
            rounding = math.floor
        else:
            rounding = math.ceil
        if target:
            times = rounding(target/(self.pyvista_mesh.n_faces*4 ))
            print(f"Subdividing {times} times")
   
        
       
        self.pyvista_mesh.subdivide(times,algo, inplace=True)
        
        
    def decimate(self,reduction=0.5,algo="pro",target=False):
            
        
        if type(self.pyvista_mesh)== bool:
            self.make_pyvista_mesh()

        if target:
            reduction = 1- target/self.pyvista_mesh.n_faces
        if reduction <= 0:
            print("Nothing to reduce")
            return
        if algo=="pro":
            self.pyvista_mesh.decimate_pro(reduction,inplace=True)
        
        else:
            self.pyvista_mesh.decimate(reduction,inplace=True)

        print(f"Decimating  {reduction}% ")
            


            

        

if __name__ == "__main__":
    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    path = Path(r"data/test.ply")
    max_path = Path('data/benchmark/db/17/m1755/m1755.off')
    problem_path = "data/benchmark/db/2/m201/m201.off"
    pig_path = Path(r"data\benchmark\db\1\m102\m102.off")
    path = path
    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    shape = Shape(vertices,element_dict,info)

    shape.make_pyvista_mesh()
    shape.subdivide(2)
    shape.process_shape()
    shape.view_processed()
    
    shape.make_pyvista_mesh_processed()
    shape.pyvista_mesh_processed.plot()
    
    print("done")


        
