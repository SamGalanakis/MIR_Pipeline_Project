import pyvista
from shape import Shape
import pyacvd
import numpy as np

def process(shape,n_faces_target=False):
    assert isinstance(shape,Shape), "Input must be instance of Shape"


    shape.make_pyvista_mesh()
    shape.pyvista_mesh.clean(inplace=True)

    
    if n_faces_target:
            
        clus = pyacvd.Clustering(shape.pyvista_mesh)

        n_subdiv= int(np.ceil(np.log(n_faces_target/clus.mesh.n_faces)/np.log(4))) # Number of subdivisions to overshoot target
        
        clus.subdivide(n_subdiv)
        clus.cluster(6000)
    
        new_mesh = clus.create_mesh()
    

        
    
        shape.pyvista_mesh = new_mesh
        shape.decimate(target=n_faces_target)
       
   
        shape.pyvista_mesh_to_base(shape.pyvista_mesh)
       
        shape.process_shape()

        return shape