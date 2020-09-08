import numpy as np


class FileReader:
    def __init__(self, path):
        self.path = path

    def convert_ply_to_off(self):
        off_file = ["OFF\n"]

        with open(self.path) as f:
            ply = f.readlines()

        ply = [x for x in ply if not x.startswith("comment")]

        vertex = ply[2].split()[2]
        indeces = ply[6].split()[2]
        off_file.append(f"{vertex} {indeces} 0\n")

        ply = ply[9:]

        off_file.extend(ply[:int(vertex)])
        off_file.extend(ply[int(vertex):])

        return off_file

    def read(self):

        if self.path.split(".")[-1] == "ply":
            lines = self.convert_ply_to_off()

        if not lines:
            with open(self.path) as f:
                lines = f.readlines()
            lines = [x for x in lines if x[0] != "#"]
        if "OFF" in lines[0]:
            lines = lines[1:]
            print("off file")

        lines = [x.rstrip() for x in lines]

        info = [int(x) for x in lines[0].split()]
        lines = lines[1:]

        if len(info) == 4:
            n_vertices = info[0]
            n_faces = info[1]
            n_edges = info[2]
            n_cells = info[3]
        else:
            n_vertices = info[0]
            n_faces = info[1]
            n_cells = info[2]

        vertices = lines[:n_vertices]
        vertices = np.array([list(map(lambda y: float(y), x.split()))
                                for x in vertices], dtype=np.float32).flatten()
        triangle_elements = lines[n_vertices:]
        triangle_elements = np.array([list(map(lambda y: int(y), x.split()))[
                                        1:] for x in triangle_elements], dtype=np.uint32).flatten()

        return vertices, triangle_elements, info
