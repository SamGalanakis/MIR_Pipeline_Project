from file_reader import read_model, write_model_as_ply
from utils import get_all_file_paths
from tqdm import tqdm

dataset_path = 'data/benchmark'


off_paths = get_all_file_paths(dataset_path,'.off')

for path in tqdm(off_paths):
    verts , faces , _ = read_model(path)
    new_path = path.replace('.off','.ply')
    write_model_as_ply(verts,faces['triangles'],new_path)
