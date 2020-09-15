import numpy as np
from pathlib import Path
from file_reader import FileReader

def loop_refinement(vertices,indices):
 





    indices = indices.reshape(-1,3)



if __name__ =="__main__":



    path = Path(r"data\\test.ply")
    reader = FileReader()
    vertices, element_dict, info = reader.read(path)
    loop_refinement(vertices,element_dict["triangles"])
