import numpy as np




def ply_to_off(path):
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






def read_file(path):
    line_list = False
    if path.suffix==".off":
        pass
    elif path.suffix==".ply":
        line_list=ply_to_off(path)
    else:
        raise Exception(f"Invalid file type: {path.suffix}")

    if not line_list:    
        with path.open() as f:
            line_list=f.readlines()
        line_list = [x for x in line_list if x[0]!= "#"]
    if "OFF" in  line_list[0]:
        line_list=line_list[1:]
        
    
    line_list= [ x.rstrip() for x in line_list]

    info= line_list[0].split()
    info = [int(x) for x in info]
    line_list = line_list[1:]
    if len(info) == 4:
        n_vertices = info[0]
        n_faces= info[1]
        n_edges= info[2]
        n_attributes = info[3]
    else:
        n_vertices = info[0]
        n_faces= info[1]
        n_attributes = info[2]

        

    if n_attributes >0:
        raise Exception("Extra properties")
    
    vertices = line_list[:n_vertices]
    vertices = np.array( [list(map(lambda y: float(y),x.split())) for x in vertices],dtype= np.float32).flatten()
    triangle_elements = line_list[n_vertices:]
    triangle_elements = np.array([list(map(lambda y: int(y),x.split()))[1:] for x in triangle_elements],dtype=np.uint32).flatten()



    return vertices, triangle_elements, info
    

         

if __name__ == "__main__":
    path = r"data\benchmark\db\0\m0\m0.off"
    path =r"data\test_ply.ply"
    linux_path =r"../../data/test_ply.ply"

    read_file(path)
