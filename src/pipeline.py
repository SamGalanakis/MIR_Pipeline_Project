import platform
from model_viewer import ModelViewer
import os
from pathlib import Path


def main():
 
    path = Path(r"data\\test.ply")

    viewer = ModelViewer()
    viewer.process(path)


if __name__ == '__main__':
    main()
