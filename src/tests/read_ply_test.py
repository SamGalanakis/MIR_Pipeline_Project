import platform
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_reader import FileReader

def test():

    if platform.system() == "Linux":
        path = r"../../data/test.ply"
    else:
        path = r"data\\test.ply"

    reader = FileReader(path)
    print(reader.read())


if __name__ == '__main__':
    test()
