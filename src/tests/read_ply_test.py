import platform
import sys
import os
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from file_reader import FileReader

def test():
   
    path = Path(r"data\\test.ply")

    reader = FileReader()
    print(reader.read(path))


if __name__ == '__main__':
    test()
