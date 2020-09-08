import numpy as np




def ply_to_off(path):
    off_file = ["OFF\n"]
    with open(path) as f:
        ply = f.readlines()
    vertex = ply[3].split()[2]
    indeces = ply[7].split()[2]
    off_file.append(f"{vertex} {indeces} 0\n")

    ply = ply[10:]

    off_file.append(ply[:int(vertex)])
    off_file.append(ply[int(vertex):])

    return off_file            






def read_file(path):
    if path.split(".")[-1]=="off":
        pass
    elif path.split(".")[-1]=="ply":
        ply_to_off(path)

    with open(path) as f:
        line_list=f.readlines()
        line_list = [x for x in line_list if x[0]!= "#"]
        if "OFF" in  line_list[0]:
            line_list=line_list[1:]
            print("off file")
        
        line_list= [ x.rstrip() for x in line_list]

        info= line_list[0].split()
        info = [int(x) for x in info]
        line_list = line_list[1:]
        if len(info) == 4:
            n_vertices = info[0]
            n_faces= info[1]
            n_edges= info[2]
            n_cells = info[3]
        else:
            n_vertices = info[0]
            n_faces= info[1]
            n_cells = info[2]
        

   
      
        vertices = line_list[:n_vertices]
        vertices = np.array( [list(map(lambda y: float(y),x.split())) for x in vertices],dtype= np.float32).flatten()
        triangle_elements = line_list[n_vertices:]
        triangle_elements = np.array([list(map(lambda y: int(y),x.split()))[1:] for x in triangle_elements],dtype=np.uint32).flatten()

   

        return vertices, triangle_elements, info
    

         

if __name__ == "__main__":
    path = r"data\benchmark\db\0\m0\m0.off"
    path =r"data\test_ply.ply"
    read_file(path)
