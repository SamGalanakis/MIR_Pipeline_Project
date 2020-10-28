from os.path import abspath, dirname, isdir, join, normpath, realpath
import sys
import pathlib
src_dir = dirname(dirname(__file__)) # for import stuff from main project
sys.path.append(src_dir)
src_dir = pathlib.Path(src_dir)


from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.plotting import curdoc
from bokeh.embed import file_html
from tornado.web import StaticFileHandler
import os.path as op
import pathlib

import sys
import bokeh.util.paths as bokeh_paths
from bokeh.models import Button
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets.inputs import FileInput
from bokeh.themes import built_in_themes
import base64

from file_reader import read_model,write_model_as_ply
#from query_interface import QueryInterface


# data_path = op.join(src_dir,"processed_data/data_processed_10000_1000000.0.csv")
# n_vertices_target = 10000
# query_interface = QueryInterface(data_path,divide_distributions=False,n_bins=10,n_vertices_target = n_vertices_target)


CODE = """
var n_models = %d;

    window.update_everything(n_models);
"""



def decode_base_64_message(base64_message):
    base64_bytes = base64_message.encode('ascii')
    message_bytes = base64.b64decode(base64_bytes)
    file_contents = message_bytes.decode('ascii')
    return file_contents

doc = curdoc()

path = 'C:/Users/samme/Google_Drive/Code_library/MIR_Pipeline_Project/ui_viewer/templates/Lucy100k.ply'
lucy_path = 'ui_viewer/static/Lucy100k.ply'
path = 'ui_viewer/static/m1405.ply'
query_path = path


run_button = Button(name = 'run_button',label="Run query!", button_type="success")
file_input = FileInput(name = 'file_input',accept='.ply,.off')

current_path = path
js_update_models_callback = CustomJS()



def file_receiver(attr, old, new):
    print('Received file!')
    input_model_path= pathlib.Path("ui_viewer/static/models/input_model.ply")
    file_contents = decode_base_64_message(new)
    file_extension = file_input.filename.split('.')[-1]
    n_models = 1

    if file_extension == 'off':
        file_contents = file_contents.splitlines()
        verts , faces , _ = read_model(file_contents)
        write_model_as_ply(verts,faces['triangles'],input_model_path)
    else:
        with open(input_model_path,'w+') as f:
            f.writelines(file_contents)
    print('File written!')
    query_path = src_dir / input_model_path
    query_return_path = input_model_path.parents[0] / 'closest_model_'
    print(query_return_path)
    #Query is made and new models are saved to the static dir
    #distances, indices, resulting_paths, resulting_classifications = query_interface.query(query_path,n_samples_query=1e+6,visualize_results=False,write_path=query_return_path)

    #Change callback of start button with appropriate number of models to be l
    js_update_models_callback.code = CODE % n_models

    



file_input.on_change('value', file_receiver)






run_button.js_on_click(js_update_models_callback)

doc.add_root(file_input)
doc.add_root(run_button)

doc.template_variables['ply_path'] = path
doc.template_variables['lucy_path'] = lucy_path




