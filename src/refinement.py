import numpy as np
from model_viewer import ModelViewer
from pathlib import Path
from file_reader import FileReader
from utils import bounding_box
from itertools import combinations 
from collections import defaultdict
from tqdm import tqdm



class SubDivision:
    def __init__(self,vertices,element_dict,info):
        self.vertices = vertices
        self.element_dict = element_dict
        self.info = info
        self.n_triangles = element_dict["triangles"].size
        self.n_quads = element_dict["quads"].size
        self.n_vertices = vertices.size
        self.get_edges()
        self.viewer = ModelViewer()

        
        self.new_subdivide()

   
    def subdivide(self):
        label_triangles = {}
        label_edges = {}
        triangle_edges = {}
        edges_triangles = defaultdict(list)

        for idx, triangle in enumerate(self.element_dict["triangles"]):
            triangle_edges[idx] = sorted([sorted(edge) for edge in list(combinations(triangle,2))])

        for idx, edge in tqdm(enumerate(self.edges)):
            for key, edges in enumerate(triangle_edges.values()):
                if edge in edges:
                    edges_triangles[idx].append(key)

        print()

    def new_subdivide(self):
        old_faces = self.element_dict["triangles"]
        old_verts = self.vertices
        edge_dict = {}

        def create_edge(indexA, indexB, face):
            a = min(indexA,indexB)
            b = max(indexA,indexB)
            key = str(a) + "~" + str(b)

            if key not in edge_dict:
                v1 = old_verts[a]
                v2 = old_verts[b]

                edge = {"vertexA" : v1, "vertexB": v2, "indexA": a, "indexB": b, "new_vert": 0, "connected_faces" : []}
                
                edge["connected_faces"].append(face)
                
                edge_dict[key] = edge

            elif key in edge_dict.keys() and face in edge_dict[key]["connected_faces"]:
                edge_dict[key]["connected_faces"].append(face)
        
        for idx, face in enumerate(old_faces):
            create_edge(face[0],face[1], idx)
            create_edge(face[1],face[2], idx)
            create_edge(face[2],face[1], idx)

        keys = list(edge_dict.keys())

        for idx, key in enumerate(keys):
            if idx + 1 == len(keys):
                break

            temp = keys[idx +1]
            faces = edge_dict[temp]["connected_faces"]

            outer_vertex = []
            a = edge_dict[temp]["indexA"]
            b = edge_dict[temp]["indexB"]

            for face in faces:
                cur_face = old_faces[face]

                if cur_face[0] != a and cur_face[1] != b:
                    outer_vertex.append(old_verts[cur_face[0]])
                elif cur_face[1] != a and cur_face[1] != b:
                    outer_vertex.append(old_verts[cur_face[b]])
                elif cur_face[2] != a and cur_face[2] != b:
                    outer_vertex.append(old_verts[cur_face[2]])

            if len(outer_vertex) == 2:
                vert1 = old_verts[a] * 3/8
                vert2 = old_verts[b] * 3/8
                vert3 = outer_vertex[0] * 1/8
                vert4 = outer_vertex[1] * 1/8
                result = vert1 + vert2 + vert3 + vert4
                print() 
            else:
                result = (old_verts[a] + old_verts[b]) / 2

            edge_dict[temp]["new_vert"] = result

            

        
        new_verts = []
        for idx, old_vert in enumerate(old_verts):
            keys_iter = iter(list(edge_dict.keys()))
            connecting_vertex = []

            for key, value in edge_dict.items():
                after = next(keys_iter)
                val1 = edge_dict[after]["indexA"]
                val2 = edge_dict[after]["indexB"]

                if val1 == idx:
                    connecting_vertex.append(val2)
                elif val2 == idx:
                    connecting_vertex.append(val1)
            
            n = len(connecting_vertex)
            if n > 3:
                beta = 3/(8*n)
            else:
                beta = 3/16
            new_vector = 0

            for i in range(n):
                new_vector += old_verts[connecting_vertex[i]] * beta

            new_vector += old_vert * (1 - (n*beta))

            new_verts.append(new_vector)
        
        old_verts = new_verts
        keys_iter = iter(list(edge_dict.keys()))

        for olf_face in old_faces:
            ab = min(face[0],face[1])
            ba = max(face[0],face[1])
            abKey = str(ab) + "~" + str(ba)
            ac = min(face[0],face[2])
            ca = max(face[0],face[2])
            acKey = str(ac) + "~" + str(ca)
            bc = min(face[1],face[2])
            cb = max(face[1],face[2])
            bcKey = str(bc) + "~" + str(cb)




    def get_edges(self):
        edges_non_unique = list(dict.fromkeys([item for t in [list(combinations(triangle,2)) for triangle in self.element_dict["triangles"]] for item in t]))
        self.edges =  np.unique([sorted(a) for a in edges_non_unique], axis = 0).tolist()
       
    def view_refined(self):
        #self.viewer.process(vertices = self.processed_vertices,indices = self.element_dict["triangles"],info=self.info)
        NotImplemented

    def view(self):
        self.viewer.process(vertices = self.vertices , indices = self.element_dict["triangles"],info=self.info)



path = Path(r"data/test.ply")
reader = FileReader()
vertices, element_dict, info = reader.read(path)
Refinement = SubDivision(vertices,element_dict,info)