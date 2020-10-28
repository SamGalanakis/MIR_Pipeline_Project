from os.path import abspath, dirname, isdir, join, normpath, realpath
import sys
import pathlib
src_dir = dirname(dirname(__file__)) # for import stuff from main project
sys.path.append(src_dir)
src_dir = pathlib.Path(src_dir)
import shutil
from bokeh.layouts import column , row
from bokeh.io import output_file,show
from bokeh.plotting import figure
from bokeh.plotting import curdoc
from bokeh.embed import file_html
from tornado.web import StaticFileHandler
import os.path as op
import pathlib
import random
import sys
import bokeh.util.paths as bokeh_paths
from bokeh.models import Button, Slider
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets.inputs import FileInput, TextInput
from bokeh.themes import built_in_themes
import pandas as pd
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
from file_reader import read_model,write_model_as_ply
#from query_interface import QueryInterface


# data_path = op.join(src_dir,"processed_data/data_processed_10000_1000000.0.csv")
# n_vertices_target = 10000
# query_interface = QueryInterface(data_path,divide_distributions=False,n_bins=10,n_vertices_target = n_vertices_target)
initial_n_neighbours = 3

CODE = """
var n_models = %d;

    window.update_everything(n_models);
"""
#Data table



columns_include=["file_name","distance","classification","n_vertices","n_triangles","n_quads",
                "volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
                "diameter","eccentricity" ]
df = pd.DataFrame(columns=columns_include)
columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns]  
data_table = DataTable(columns=columns, source=ColumnDataSource(df),height = 150,sizing_mode = "stretch_width",name = "data_table") 




doc = curdoc()


slider_n_neighbours = Slider(start=1, end=15, value=initial_n_neighbours, step=1, title="Number of neighbours ",name="slider_n_neighbours")
run_button = Button(name = 'run_button',label="Run query!", button_type="success")

js_update_models_callback = CustomJS(code = (CODE % initial_n_neighbours))

path_input = TextInput(name='path_input')

def slider_n_neighbours_callback(attr,old,new):
    initial_n_neighbours = int(new)
    js_update_models_callback.code = CODE % initial_n_neighbours
    print(f'slider_n_neighbours:{initial_n_neighbours}')
    

def path_callback(attr, old, new):
    print(f"New path {new}")
    input_model_static_path= pathlib.Path("ui_viewer/static/models/input_model.ply")
    query_return_path = input_model_static_path.parents[0] / 'closest_model_'
    input_path = pathlib.Path(new)
    shutil.copyfile(input_path, input_model_static_path)
    #distances, indices, resulting_paths, resulting_classifications,df_slice = query_interface.query(input_path,n_samples_query=1e+6,visualize_results=False)
    distances.insert(0,0)
    df_slice['distance'] = distances
    df = df_slice[columns_include]

    #change to ply target
    
    resulting_paths = [x.replace('.off','.ply') for x in resulting_paths]
    #copy files to static
    for index, model_match_path in enumerate(resulting_paths):
        shutil.copyfile(model_match_path,f"{str(query_return_path)}{index}.ply")

path_input.on_change('value',path_callback)





slider_n_neighbours.on_change('value',slider_n_neighbours_callback)




run_button.js_on_click(js_update_models_callback)


options_col = column(path_input,slider_n_neighbours,run_button,name = 'options_col')
full_row = row(options_col,data_table,name= 'full_row')
# doc.add_root(run_button)
# doc.add_root(path_input)
# doc.add_root(data_table)
# doc.add_root(slider_n_neighbours)
doc.add_root(full_row)





