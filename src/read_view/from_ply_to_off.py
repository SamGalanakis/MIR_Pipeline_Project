import numpy as np





def ply_to_off(path):
    off_file = ["OFF\n"]
    with open(path) as f:
        ply = f.readlines()
        vertex = ply[3].split()[2]
        indeces = ply[7].split()[2]
        off_file.append(f"{vertex} {indeces} 0\n")
        ply = ply[10:]
        a = ply[int(vertex)-1:]

        off_file.append(ply[:int(vertex)])
        off_file.append(ply[int(vertex):])

        return off_file
        #for line in ply:
            





path = "/home/mvalente/docs/msc_ai/year2/term1/informr/MIR_Pipeline_Project/data/m0.ply"

ply_to_off(path)