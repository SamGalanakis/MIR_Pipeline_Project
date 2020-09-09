from file_reader import FileReader
from model_viewer import ModelViewer
import pathlib
import numpy as np



path = pathlib.Path(r"data\test.ply")
reader = FileReader()
viewer = ModelViewer()
vertices, indices, info = reader.read(path)


viewer.process(path)
# viewer.process(vertices=vertices,indices=indices)


as_points = vertices.reshape(-1, 3)

mean_point = as_points.mean(axis=0)

max_x, max_y, max_z = as_points.max(axis=0)
min_x, min_y, min_z = as_points.min(axis=0)

middle_point = np.array(
    [min_x + (max_x-min_x)/2, min_y + (max_y-min_y)/2, min_z + (max_z-min_z)/2])

bounding_rect_vertices = np.array([[min_x,min_y,min_z],[max_x,min_y,min_z],[max_x,min_y,max_z],[min_x,min_y,max_z],
                            [min_x,max_y,min_z],[max_x,max_y,min_z],[max_x,max_y,max_z],[min_x,max_y,max_z]]).flatten()


bounding_rect_indices = (np.array([0,1,2,2,3,0,  4,5,6,6,7,4, 0,1,4,4,5,1,  2,3,6,6,7,3, 0,3,4,4,7,3, 1,2,5,5,6,2 ],dtype=np.uint32) + max(indices)+1).flatten()

vertices = np.append(vertices,bounding_rect_vertices)

normalized_vertices = vertices/max(abs(vertices))
indices = np.append(indices,bounding_rect_indices)




