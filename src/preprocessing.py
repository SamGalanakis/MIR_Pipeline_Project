import pyvista
from shape import Shape
import pyacvd
import numpy as np

def process(shape,n_vertices_target=False):
    assert isinstance(shape,Shape), "Input must be instance of Shape"


    shape.make_pyvista_mesh()
    shape.pyvista_mesh.clean(inplace=True)

    
       
    
    if n_vertices_target:
            
        clus = pyacvd.Clustering(shape.pyvista_mesh)
        n_subdiv= int(np.ceil(np.log(n_vertices_target/clus.mesh.n_points)/np.log(4))) # Number of subdivisions to overshoot target
        
        if n_subdiv>0:
            clus.subdivide(n_subdiv)
        clus.cluster(n_vertices_target)

        new_mesh = clus.create_mesh()
        shape.pyvista_mesh = new_mesh
    
        shape.pyvista_mesh_to_base(shape.pyvista_mesh)
    
           
       
    shape.process_shape()

    return shape