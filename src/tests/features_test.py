import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_reader import FileReader
from model_viewer import ModelViewer
from shape import Shape
from pathlib import Path
from utils import bounding_box, cla_parser, calculate_diameter, align, angle_three_vertices, barycenter_vertice, two_vertices, cube_volume_tetrahedron, barycenter_vertice,square_area_triangle







def test_square_area_triangle(vertices):
    a = np.array([1,-1,1])
    b = np.array([0,1,1])
    c = np.array([1,1,1])

    assert np.sqrt(1/2 * np.linalg.norm((a[0]-c[0])*(b[1]-a[1]) - (a[0] - b[0])*(c[1]-a[1]))) == 1 , "square_area_triangle() failed the test"






if __name__ == '__main__':
    path = Path(r"data/cube.off")
    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    shape = Shape(vertices,element_dict,info)


    
    test_square_area_triangle(vertices)
    print("------------- test_square_area_triangle() passed the test")
