import numpy as np







def read_file(path):
    if path.split(".")[-1]=="off":
        pass
    with open(path) as f:
        line_list=f.readlines()
        if "OFF" in  line_list[0]:
            line_list=line_list[1:]
            print("off file")
        
        line_list= [ x.rstrip() for x in line_list]

        info= line_list[0].split(" ")
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
        vertices = np.array( [list(map(lambda y: float(y),x.split(" "))) for x in vertices],dtype= np.float32).flatten()
        triangle_elements = line_list[n_vertices:]
        triangle_elements = np.array([list(map(lambda y: int(y),x.split(" ")))[1:] for x in triangle_elements],dtype=np.uint32).flatten()

        print("readfile")

        return vertices, triangle_elements, info
    

         

if __name__ == "__main__":
    path = r"data\benchmark\db\0\m0\m0.off"
    read_file(path)
