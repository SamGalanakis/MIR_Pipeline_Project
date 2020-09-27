import platform
from model_viewer import ModelViewer
from database_creator import Database
import os
from pathlib import Path


def main():
 
    path = Path(r"data/test.ply")

    #viewer = ModelViewer()
    #viewer.process(path)

    database = Database()
    database.create_database("processed_data", True)



if __name__ == '__main__':
    main()
