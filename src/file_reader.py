import numpy as np
from pathlib import Path





def write_model_as_ply(verts,faces,path):
    verts = verts.reshape((-1,3))
    faces = faces.reshape((-1,3))
    n_verts = verts.shape[0]
    n_faces = faces.shape[0]

    ply_lines = ['ply','format ascii 1.0',f'element vertex {n_verts}','property float x','property float y'
    ,'property float z',f'element face {n_faces}', 'property list uchar int vertex_indices','end_header']
    
    faces = np.hstack((np.ones((faces.shape[0],1),dtype=np.int)*faces.shape[1],faces))
    
    faces = [[str(x) for x in y] for y in faces.tolist()]
    faces = [' '.join(x) for x in faces]

    verts = [[str(x) for x in y] for y in verts.tolist()]
    verts = [' '.join(x) for x in verts]

    ply_lines.extend(verts)
    ply_lines.extend(faces)
    ply_lines = [ x+'\n' for x in ply_lines]
    with open(path,'w+') as f:
        f.writelines(ply_lines)
    






        
def convert_ply_to_off(path):
    off_file = ["OFF\n"]

    with path.open() as f:
        ply = f.readlines()

    ply = [x for x in ply if not x.startswith("comment")]

    vertex = ply[2].split()[2]
    indeces = ply[6].split()[2]
    off_file.append(f"{vertex} {indeces} 0\n")

    ply = ply[9:]

    off_file.extend(ply[:int(vertex)])
    off_file.extend(ply[int(vertex):])

    return off_file

def read_model(path,verbose=False):
    '''Read model file, off or ply '''
    if not type(path)==list:
        if type(path)==str:
            path = Path(path)
        lines=False
        if path.suffix == ".ply":
            lines = convert_ply_to_off(path)
        elif path.suffix != ".off":
            raise Exception("Invalid file type, can only process .off and .ply")

        if not lines:
            with path.open() as f:
                lines = f.readlines()
    else:
        lines = path
        lines = [x for x in lines if x[0] != "#"]
    if "OFF" in lines[0]:
        lines = lines[1:]
        

    lines = [x.rstrip() for x in lines]

    info = [int(x) for x in lines[0].split()]
    lines = lines[1:]

    if len(info) == 4:
        n_vertices = info[0]
        n_faces = info[1]
        n_edges = info[2]
        n_attributes = info[3]
    else:
        n_vertices = info[0]
        n_faces = info[1]
        n_attributes = info[2]

    if n_attributes > 0:
        raise Exception("Extra properties")

    vertices = lines[:n_vertices]
    vertices = np.array([list(map(lambda y: float(y), x.split()))
                            for x in vertices], dtype=np.float32).flatten()
    elements = lines[n_vertices:]
    elements = [list(map(lambda y: int(y), x.split())) for x in elements]

    triangles = np.array([x[1:] for x in elements if x[0]==3],dtype = np.uint32)
    quads = np.array([x[1:] for x in elements if x[0]==4],dtype = np.uint32)
    assert triangles.size/3 +quads.size/4 == len(elements), "Non quad/triangle elements!"
    element_dict = {"triangles":triangles, "quads":quads}
    if verbose:
        print(f" File type: {path.suffix} Triangles: {triangles.size}, Quads: {quads.size}.")
    return vertices, element_dict, info

if __name__ == '__main__':
    
    verts , faces , _ = read_model('data/benchmark/db/1/m117/m117.off')
    out = write_model_as_ply(verts,faces['triangles'],'new.ply')
    print('')