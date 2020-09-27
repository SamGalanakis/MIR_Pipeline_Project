import platform
from model_viewer import ModelViewer
from database_creator import Database
import os
from pathlib import Path


def main():
 
    path = Path(r"data/test.ply")

    viewer = ModelViewer()
    viewer.process(path)

    database = Database()
    database.create_database(False, "pre_processed_data")



if __name__ == '__main__':
    main()
