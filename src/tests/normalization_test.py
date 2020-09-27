import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os
from shape import Shape

from pathlib import Path
from file_reader import FileReader

def test_barycenter(shape):
    assert abs(shape.processed_vertices.mean(axis=0).max())<1e-5, "Not translated to barycenter!"

def test_alignment(shape):
    vertices = shape.processed_vertices.flatten()
    vertices = vertices.reshape((3,-1),order="F")
    cov = np.cov(vertices)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    assert (abs(eigenvectors.sum(axis=0) -1) <1e-5).sum()==3, "Eigenvectors are not aligned with standard basis!"

def test_normalization(shape):
    vertices = shape.processed_vertices
    pass





if __name__ == '__main__':

    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    path = Path(r"data/test.ply")
    max_path = Path('data/benchmark/db/17/m1755/m1755.off')
    problem_path = "data/benchmark/db/2/m201/m201.off"
    path = ant_path



    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    shape = Shape(vertices,element_dict,info)
    shape.make_pyvista_mesh_processed()

    test_barycenter(shape)
    test_alignment(shape)
    
