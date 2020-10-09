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
        targ = 100000
        n_subdiv= int(np.ceil(np.log(targ/clus.mesh.n_faces)/np.log(4))) # Number of subdivisions to overshoot target
        
        if n_subdiv>0:
            clus.subdivide(n_subdiv)
        clus.cluster(n_faces_target/2)

        new_mesh = clus.create_mesh()
        shape.pyvista_mesh = new_mesh
       # shape.decimate(target=n_faces_target)
        shape.pyvista_mesh_to_base(shape.pyvista_mesh)
        diff = 0.1*n_faces_target
        if np.abs(shape.pyvista_mesh.n_faces-n_faces_target)>diff: 
            print(f'Shape not within {diff} of target, distance is {shape.pyvista_mesh.n_faces-n_faces_target}')
           
       
    shape.process_shape()

    return shape