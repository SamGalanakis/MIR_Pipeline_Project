from pathlib import Path
import numpy as np
import torch
from torch.utils.data import  Dataset, DataLoader
import pytorch3d

from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from utils import cla_parser, merge_dicts, get_all_file_paths
from shape import Shape
from preprocessing import process
from file_reader import FileReader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pytorch3d.ops import sample_points_from_meshes
from typing import Dict, List

if torch.cuda.is_available():
    device = torch.device("cuda:0")


def plot_pointcloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = Axes3D(fig)
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

class princetonDataSet(Dataset):
    def  __init__(self,file_list,n_vertices_target):
        self.n_vertices_target=n_vertices_target
        self.file_list = file_list
        self.reader = FileReader()

    
    def __len__(self):
        return len(self.file_list)


    def __getitem__(self, idx): 
        path = self.file_list[idx]
        shape = Shape(*self.reader.read(path))
        shape = process(shape,n_vertices_target=self.n_vertices_target)
        shape.pyvista_mesh_to_base(shape.pyvista_mesh)
        vertices = shape.vertices
        faces = shape.element_dict['triangles']

        verts =  torch.FloatTensor(vertices)
        faces  = torch.LongTensor(faces.astype(np.int32))
        mesh = Meshes(faces=[faces],verts=[verts.reshape((-1,3))])

        return mesh

if __name__ == '__main__':
    dolphin_path = 'data/dolphin.obj'
    verts, faces, aux = load_obj(dolphin_path)
    file_list =  get_all_file_paths(r'data/benchmark','.off')
    dataset = princetonDataSet(file_list,n_vertices_target=10000)
    dataloader = DataLoader(dataset,batch_size=10)
    mesh  = dataset[1]
    plot_pointcloud(mesh, "Target mesh")
    plt.show()
    print(verts)
    

