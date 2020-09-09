from file_reader import FileReader
from model_viewer import ModelViewer
import pathlib



path = pathlib.Path(r"data\test.ply")
reader = FileReader()
viewer = ModelViewer()
vertices, indices, info = reader.read(path)

viewer.process(vertices=vertices,indices=indices)
