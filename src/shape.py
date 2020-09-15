import numpy as np
from model_viewer import ModelViewer
from pathlib import Path
from file_reader import FileReader
from utils import bounding_box
from itertools import combinations 



class Shape:
    def __init__(self,vertices, element_dict, info):
        self.vertices=vertices
        self.element_dict = element_dict
        self.info = info
        self.n_triangles = element_dict["triangles"].size
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size
        self.viewer = ModelViewer()
        self.get_edges()
        self.n_edges = len(self.edges)
        

        self.barycenter =   vertices.reshape(-1, 3).mean(axis=0) #is this barycenter or just centroid??

        self.normalized_barycentered_vertices = vertices.reshape(-1, 3).mean(axis=0) - self.barycenter

        self.process_shape()

    def get_edges(self):
        self.edges = set() 
        edges_non_unique = list(dict.fromkeys([item for t in [list(combinations(triangle,2)) for triangle in self.element_dict["triangles"]] for item in t]))

        for (a, b) in edges_non_unique:
            if (a, b) and (b ,a) not in self.edges:
                self.edges.add((a,b))

    def process_shape(self):
        self.barycenter =   self.vertices.reshape(-1, 3).mean(axis=0)
        processed_vertices = self.vertices.reshape(-1, 3) - self.barycenter
    
        self.processed_vertices = (processed_vertices- processed_vertices.min(axis=0))/(processed_vertices.max(axis=0)- processed_vertices.min(axis=0)).flatten()
        self.bounding_rect_vertices, self.bounding_rect_indices = bounding_box(self.processed_vertices,self.element_dict["triangles"])

    def view_processed(self):
        self.viewer.process(vertices = self.processed_vertices,indices = self.element_dict["triangles"],info=self.info)
    def view(self):
        self.viewer.process(vertices = self.vertices , indices = self.element_dict["triangles"],info=self.info)






if __name__ == "__main__":
    path = Path(r"data\\test.ply")
    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    shape = Shape(vertices,element_dict,info)
    shape.view_processed()
    print("done")


        
