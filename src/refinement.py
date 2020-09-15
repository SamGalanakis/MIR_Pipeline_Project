import numpy as np
from model_viewer import ModelViewer
from pathlib import Path
from file_reader import FileReader
from utils import bounding_box
from itertools import combinations 


class Refinement:
    def __init__(self,vertices,element_dict,info):
        self.vertices = vertices
        self.element_dict = element_dict
        self.info = info
        self.triangles = element_dict["triangles"]
        self.n_triangles = element_dict["triangles"].size
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size
        self.viewer = ModelViewer()

        
        self.refine()

    def refine(self):
        #for every edge in the source mesh add a vertex
        #for every triangle on the mesh create four triangles  
        ##The geometric location of both above is determined by a subdivision scheme
        average_faces = [int(sum(triangle)/len(triangle)) for triangle in self.triangles]
       
        #compute edges
        #edges_per_face = [list(combinations(triangle,2)) for triangle in self.triangles]
        edges_non_unique = list(dict.fromkeys([item for t in [list(combinations(triangle,2)) for triangle in self.triangles] for item in t]))
        edges_indice = set() 

        for (a, b) in edges_non_unique:
            if (a, b) and (b ,a) not in edges_raw:
                edges_raw.add((a,b))

        for edge in edges:
             

        print()

    def view_refined(self):
        #self.viewer.process(vertices = self.processed_vertices,indices = self.element_dict["triangles"],info=self.info)
        NotImplemented

    def view(self):
        self.viewer.process(vertices = self.vertices , indices = self.element_dict["triangles"],info=self.info)



path = Path(r"data/cube.off")
reader = FileReader()
vertices, element_dict, info = reader.read(path)
Refinement = Refinement(vertices,element_dict,info)