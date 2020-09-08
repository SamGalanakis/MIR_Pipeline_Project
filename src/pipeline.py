import platform
from model_viewer import ModelViwer
import os


def main():
    if platform.system() == "Linux":
        os.system('export MESA_GL_VERSION_OVERRIDE=3.3')
        path = r"../data/test.ply"
    else:
        path = r"data\\test.ply"

    viewer = ModelViwer()
    viewer.process(path)


if __name__ == '__main__':
    main()
