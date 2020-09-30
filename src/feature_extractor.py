
from shape import Shape
from file_reader import FileReader
from preprocessing import process
from pathlib import Path


def extract_features(shape,n_faces_target):
    '''Input must be Shape or path'''
    if isinstance(shape,Shape):
        pass
    elif isinstance(shape,Path) or isinstance(shape,str):
        shape = Path(shape)
        reader = FileReader()
        shape = Shape(*reader.read(shape))
    else:
        raise Exception("Input must be Shape or path")

    shape = process(shape,n_faces_target=n_faces_target)

    feature_names=["file_name","n_vertices","n_triangles","n_quads","bounding_box",
            "volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
            "diameter","eccentricity", "angle_three_vertices","barycenter_vertice", "two_vertices",
            "square_area_triangle", "cube_volume_tetrahedron" ]



    feature_dict = {k:[] for k in feature_names}

    return feature_dict

if __name__=='__main__':
    path = Path(r"data/test.ply")
    extract_features(path)