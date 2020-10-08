import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_reader import FileReader
from model_viewer import ModelViewer
from shape import Shape
from pathlib import Path
from utils import bounding_box, cla_parser, calculate_diameter, align, angle_three_random_vertices, barycenter_vertice, two_vertices, cube_volume_tetrahedron, barycenter_vertice,square_area_triangle
from feature_extractor import extract_features




def test_volume(volume):
    assert np.abs(volume - 8 ) < 10e-3, "Volume test failed"

def test_surface_area(surface_area):
    assert np.abs(surface_area - 4*6 ) < 10e-3, "Area test failed"

def test_bounding_box_volume(bounding_box_volume):
    #Bounding box volume should be same as volume for cube
    assert np.abs(bounding_box_volume - 8 ) < 10e-3, "Bounding box volume test failed"

def test_bounding_box_ratio(bounding_box_ratio):
    #Ratio should be 1 since bounding box is cube
    assert np.abs(bounding_box_ratio - 1 ) < 10e-3, "Bouding box ratio test failed"
def test_diameter(diameter):
        #Cube is already convex so diameter just dist of opposing corners
        dist = np.linalg.norm(np.array([-1,-1,-1])- np.array([1,1,1]))
        assert np.abs(dist - diameter ) < 10e-3, "Dimateter test failed"










if __name__ == '__main__':
    #cube. off is a 2x2x2 cube
    path = Path(r"data/cube.off")
    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    shape = Shape(vertices,element_dict,info)
    feature_dict = extract_features(shape)
    test_volume(feature_dict['volume'])
    test_surface_area(feature_dict['surface_area'])
    test_bounding_box_volume(feature_dict['bounding_box_volume'])
    test_bounding_box_ratio(feature_dict['bounding_box_ratio'])
    test_diameter(feature_dict['diameter'])






