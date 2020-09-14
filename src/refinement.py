import numpy as np
from model_viewer import ModelViewer
from pathlib import Path
from file_reader import FileReader
from utils import bounding_box


class Refinement:
    def __init__(self,vertices,element_dict,info):
        self.vertices=vertices
        self.element_dict = element_dict
        self.info = info
        self.n_triangles = element_dict["triangles"].size
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size
        self.viewer = ModelViewer()

    def refine(self):

        NotImplemented
    def view_refined(self):
        #self.viewer.process(vertices = self.processed_vertices,indices = self.element_dict["triangles"],info=self.info)
        NotImplemented

    def view(self):
        self.viewer.process(vertices = self.vertices , indices = self.element_dict["triangles"],info=self.info)

