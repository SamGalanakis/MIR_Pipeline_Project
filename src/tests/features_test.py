import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_reader import FileReader
from model_viewer import ModelViewer
from shape import Shape
from pathlib import Path
from utils import bounding_box, cla_parser, calculate_diameter, align, angle_three_vertices, barycenter_vertice, two_vertices, cube_volume_tetrahedron, barycenter_vertice,square_area_triangle


def test_angle_three_vertices(vertices):
    a = np.array([1,-1,1])
    b = np.array([0,1,1])
    c = np.array([1,1,1])
    ba = a - b
    bc = c - b
    bb= np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    bbb = np.degrees(np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)))
    assert np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc)) == 0.4472135954999579, "test_angle_three_vertices() failed the test"


def test_barycenter_vertice(vertices, barycenter):
    possible_value = 1.7320508

    results = barycenter_vertice(vertices,barycenter)
        
    for result in results:
        assert results != possible_value , "test_barycenter_vertice() failed the test"
  
    print("------------- test_barycenter_vertice() passed the test")

def test_square_area_triangle(vertices):
    a = np.array([1,-1,1])
    b = np.array([0,1,1])
    c = np.array([1,1,1])

    assert np.sqrt(1/2 * np.linalg.norm((a[0]-c[0])*(b[1]-a[1]) - (a[0] - b[0])*(c[1]-a[1]))) == 1 , "square_area_triangle() failed the test"


def test_cube_volume_tetrahedron(vertices):
    possible_values =  [0,1.100642416298209]

    for i in range(4):
        _ , volumes = cube_volume_tetrahedron(vertices)

        for volume in volumes:
            assert volume in possible_values, "test_cube_volume_tetrahedron() failed the test"

    print("------------- test_cube_volume_tetrahedron() passed the test")


def grouped(iterable, n):
    return zip(*[iter(iterable)]*n)



if __name__ == '__main__':
    path = Path(r"data/cube.off")
    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    shape = Shape(vertices,element_dict,info)

    test_angle_three_vertices(vertices)
    print("------------- test_angle_three_vertices()) passed the test")

    test_barycenter_vertice(vertices,shape.barycenter)

    test_cube_volume_tetrahedron(vertices)
    
    test_square_area_triangle(vertices)
    print("------------- test_square_area_triangle() passed the test")
