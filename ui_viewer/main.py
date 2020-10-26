from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.plotting import curdoc
from bokeh.embed import file_html
from tornado.web import StaticFileHandler
import os.path as op
import pathlib
from os.path import abspath, dirname, isdir, join, normpath, realpath
import sys
import bokeh.util.paths as bokeh_paths
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets.inputs import FileInput
from bokeh.themes import built_in_themes
import base64
from ..src import file_reader
def decode_base_64_message(base64_message):
    base64_bytes = base64_message.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)
    file_contents = message_bytes.decode('ascii')
    return file_contents

doc = curdoc()

print(bokeh_paths.ROOT_DIR)

print(bokeh_paths.serverdir())



file_input = FileInput(name = 'file_input',accept='.ply,.off')

print(file_input.name)
def file_receiver(attr, old, new):
    path_to_write_to="inputted_file"
    file_contents = decode_base_64_message(new)
    file_extension = file_input.filename.split('.')[-1]
    print(file_contents.splitlines()[0:10])
    if file_extension == 'off':
        file_contents = file_contents.splitlines()
        verts , faces , _ = file_reader.read_model(file_contents)

        file_reader.write_model_as_ply(verts,faces['triangles'],f'{path_to_write_to}.ply')
    else:
        with open(f'{path_to_write_to}.ply','w+') as f:
            f.writelines(file_contents)

    
    


file_input.on_change('value', file_receiver)

doc.add_root(file_input)

path = 'C:/Users/samme/Google_Drive/Code_library/MIR_Pipeline_Project/ui_viewer/templates/Lucy100k.ply'
path = 'ui_viewer/static/Lucy100k.ply'
path = 'ui_viewer/static/m1405.ply'
doc.template_variables['ply_path'] = path




