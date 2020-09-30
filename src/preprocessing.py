import pyvista
from shape import Shape


def process(shape,n_faces_target=False):
    assert isinstance(shape,Shape), "Input must be instance of Shape"


    shape.make_pyvista_mesh()
    shape.pyvista_mesh.clean(inplace=True)


    if n_faces_target:
            
        # try:
        #     shape.subdivide(target=n_faces_target,undercut=False)
        #     if shape.pyvista_mesh.n_faces==0:
        #         raise Exception
        # except:
            
        #     print(f"Could not subdivie {file}")
        #    
        #     continue
    
        try:
            shape.decimate(target=n_faces_target)
            if shape.pyvista_mesh.n_faces==0:
                raise Exception
        except:
            print(f"Could not decimate")
        shape.pyvista_mesh_to_base(shape.pyvista_mesh)

        shape.process_shape()

        return shape