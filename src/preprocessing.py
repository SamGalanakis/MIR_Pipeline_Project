import pyvista
from shape import Shape
import pyacvd
import numpy as np
from file_reader import read_model
from pathlib import Path

from model_viewer import ModelViewer


def process(shape,n_vertices_target=False):
    assert isinstance(shape,Shape), "Input must be instance of Shape"

    
    shape.make_pyvista_mesh()
    shape.pyvista_mesh.clean(inplace=True)

    
       
    
    if n_vertices_target:
            
        clus = pyacvd.Clustering(shape.pyvista_mesh)
       # target = 4 * n_vertices_target  #Suvdivide to some larger than target so we can cluster down
     #   n_subdiv= int(np.ceil(np.log(target/clus.mesh.n_points)/np.log(4))) # Number of subdivisions to overshoot target
        
        try:
            while len(clus.mesh.points) < 30000:
                clus.subdivide(2)
        except:
            print("MEMERY!!!")
        clus.cluster(n_vertices_target)

        new_mesh = clus.create_mesh()
        ##
        


        ##
        distance_from_target  = np.abs(n_vertices_target-new_mesh.n_points)
        if distance_from_target>100:
            print(f"Distance: {distance_from_target}")
        shape.pyvista_mesh = new_mesh
        
        shape.pyvista_mesh_to_base(new_mesh)
    
    shape.process_shape()
       
    

    return shape


if __name__=='__main__':
    
    ant_path = Path(r"data/benchmark/db/0/m0/m0.off")
    path = Path(r"data/test.ply")
    
    max_path = Path('data/benchmark/db/17/m1755/m1755.off')
    problem_path = "data/benchmark/db/2/m201/m201.off"
    pig_path = Path(r"data\benchmark\db\1\m102\m102.off")
    path = pig_path
    
  
    vertices, element_dict, info = read_model(path)
    shape = Shape(vertices,element_dict,info)

    #shape.view()
    shape.make_pyvista_mesh()
    shape.pyvista_mesh_to_base(shape.pyvista_mesh)
    process(shape,n_vertices_target=10000)