from os.path import abspath, dirname, isdir, join, normpath, realpath
import os
import sys
import pathlib
from bokeh.models.widgets.widget import Widget
import numpy as np
import glob
from bokeh.models.annotations import Title
from bokeh.models import ColumnDataSource, HoverTool, LinearColorMapper
src_dir = dirname(dirname(__file__)) # for import stuff from main project
sys.path.append(src_dir)
src_dir = pathlib.Path(src_dir)
import shutil
import random
from bokeh.layouts import column , row
import matplotlib.pyplot as plt
from bokeh.plotting import figure,curdoc,show,save
import os.path as op
import pathlib
import random
import sys
import bokeh.util.paths as bokeh_paths
from bokeh.models import Button, Slider, Dropdown
from bokeh.models.callbacks import CustomJS
from bokeh.models.widgets.inputs import FileInput, TextInput
from bokeh.themes import built_in_themes
import pandas as pd
from bokeh.models import ColumnDataSource, DataTable, DateFormatter, TableColumn
from file_reader import read_model,write_model_as_ply
from query_interface import QueryInterface
from utils import  is_array_col
import matplotlib as mpl
from bokeh.transform import factor_cmap
from bokeh.palettes import Spectral6
import colorcet as cc
from make_tsne import make_tsne
#bokeh serve --show ui_viewer --address 0.0.0.0 --port=8000 --allow-websocket-origin=35.204.195.143:8000 --allow-websocket-origin=localhost:8000


data_path = op.join(dirname(src_dir),"processed_data/data_coarse1_processed_10000_10000000.0.csv")
n_vertices_target = 10000
query_interface = QueryInterface(data_path,divide_distributions=True,n_bins=10,n_vertices_target = n_vertices_target)
initial_n_neighbours = 3
tsne_path = op.join(dirname(src_dir),'processed_data/tsne_data.csv')

CODE = """
var n_models = %d;
var rand_base = %d;

    window.update_everything(n_models,rand_base);
"""

tsne_figure = make_tsne(tsne_path)
def make_data_sources_distributions(distribution_columns,df):
    data_list = {x:[] for x in distribution_columns}
    for distribution_feature in distribution_columns:
        relevant_cols = [x for x in df.columns if is_array_col(distribution_columns,x)==distribution_feature ]
        for index , row in df.iterrows():
            classification = row['classification']
            name = op.basename(row['file_name'].replace('\\','/'))
            y= row[relevant_cols].values
            x = range(0,len(y))
            data_list[distribution_feature].append({f"x_values":x,
            f"y_values":y,'name':name,'classification':classification})
    return data_list
            
def make_vacant_col(plot_col,df,distribution_columns):
    
    rows = [figure(plot_height=100,plot_width=300,tools=[]) for x in range(df.shape[0])]
    for index,row_fig in enumerate(rows):
        row = df.iloc[index,:]
        name = op.basename(row['file_name'].replace('\\','/'))
        classification = row['classification']
        source = ColumnDataSource({'x_values':[],'y_values':[]})
        title = Title()
        title.text = f'{name} ({classification})'
        row_fig.title = title

        l=row_fig.line(x='x_values', y='y_values', source=source,name='lineplot')
        row_fig.toolbar.logo =None
        
    plot_col.children.extend(rows)

    return  plot_col


def updade_plot_col(plot_col,data_list,distribution_name):
    new_data_list = data_list[distribution_name]
    for index,child in enumerate(plot_col.children):
        
        line_plot = child.select(name='lineplot')
        
      
        
        data_for_plot = {'x_values':new_data_list[index]['x_values'],'y_values':new_data_list[index]['y_values']}
        
        line_plot.data_source.data = data_for_plot
    return plot_col







    

columns_include=["file_name","distance","classification","n_triangles",
                "volume","surface_area","bounding_box_ratio","compactness","bounding_box_volume",
                "diameter","eccentricity" ]

distribution_columns = [ 'angle_three_vertices', 'barycenter_vertice',
 'two_vertices', 'square_area_triangle', 'cube_volume_tetrahedron']      

menu = list(zip(distribution_columns,distribution_columns))
distribution_dropdown = Dropdown(label="Distribution to plot", button_type="warning", 
menu=menu,name="distribution_dropdown",css_classes=['dropdown'])
        
df = pd.DataFrame(columns=columns_include)


columns = [TableColumn(field=Ci, title=Ci) for Ci in df.columns] 
data_table_source = ColumnDataSource({col:[] for col in columns_include}) 
data_table = DataTable(columns=columns, source=data_table_source,height = 150,sizing_mode = "stretch_width",name = "data_table")






doc = curdoc()


slider_n_neighbours = Slider(start=0, end=30, value=initial_n_neighbours, step=1,
 title="Number of neighbours ",name="slider_n_neighbours")
run_button = Button(name = 'run_button',label="Visualize results!", button_type="success")
rand_base=123
js_update_models_callback = CustomJS(code = (CODE % (initial_n_neighbours,rand_base)))

path_input = TextInput(name='path_input')

def distribution_dropdown_callback(event):
    global plot_col
    if not 'data_list' in globals():
        print('No query set soo...')
        return

   
    plot_col = updade_plot_col(plot_col,data_list,event.item)
distribution_dropdown.on_click(distribution_dropdown_callback)
def slider_n_neighbours_callback(attr,old,new):
    global initial_n_neighbours 
    initial_n_neighbours = int(new)
    global rand_base
    js_update_models_callback.code = CODE % (initial_n_neighbours,rand_base)
    
    
def empty_directory(dir_path):
    print('deleting files!')
    assert 'static' in str(dir_path), 'You sure you want to delete that?'
    files = glob.glob(str(dir_path))
    for f in files:
        os.remove(f)
empty_directory(pathlib.Path("ui_viewer/static/models/*"))

def path_callback(attr, old, new):
    print(f"New path {new}")
    global rand_base
    rand_base =random.randint(0,10000000)
    js_update_models_callback.code = CODE % (initial_n_neighbours,rand_base)

    
    input_model_static_path= pathlib.Path(f"ui_viewer/static/models/{rand_base}input_model.ply")
    query_return_path = input_model_static_path.parents[0] / f'{rand_base}closest_model_'
    input_path = pathlib.Path(new)

    suffix = input_path.suffix 
    
    if suffix == '.ply':
        shutil.copyfile(input_path, input_model_static_path)
    elif suffix == '.off':
        vertices , faces_dict , _ = read_model(input_path)
        write_model_as_ply(vertices,faces_dict['triangles'],input_model_static_path)
    else:
        print('Invalid file type!')
        return

    _, _ , df_slice = query_interface.query(input_path,n_samples_query=10e+6,n_results=initial_n_neighbours)
    
    
   
    df_slice = df_slice.round(4)
    global data_table
    global data_table_source
    global columns_include
    # Store paths for js, remove first as it is query path and dealing with it seperately
    paths_for_js = [x.replace('.off','.ply') for x in df_slice['file_name']][1:]
    df_slice['file_name'] = df_slice['file_name'].apply(lambda x:op.basename(x.replace('\\','/')))
    update_dict = {col:df_slice[col].tolist() for col in columns_include}
   
    
    global plot_col
    plot_col.children = []
    plot_col_template = make_vacant_col(plot_col,df_slice,distribution_columns)
    
    global data_list
    data_list = make_data_sources_distributions(distribution_columns,df_slice)
    
    plot_col = updade_plot_col(plot_col_template,data_list,distribution_columns[0])
    
 
    data_table.source.data = update_dict
    
    
    
    #change to ply target
    
    
    #copy files to static
   
    for index, model_match_path in enumerate(paths_for_js):
   

        
        
       
        model_match_path = pathlib.Path(model_match_path)
        model_match_path = src_dir.parents[0] / model_match_path.__str__().replace('\\','/')
        
        write_to_path = op.join(src_dir,f"{str(query_return_path)}{index}.ply")
        
       
        new_path = shutil.copyfile(model_match_path,write_to_path)
        print(f"Querying: {new_path}")
    




path_input.on_change('value',path_callback)





slider_n_neighbours.on_change('value',slider_n_neighbours_callback)




run_button.js_on_click(js_update_models_callback)


options_col = column(path_input,slider_n_neighbours,run_button,name = 'options_col',height=150,width= 300)
full_row = row(options_col,data_table,name= 'full_row',css_classes=['full_row'],height= 150)

# doc.add_root(run_button)
# doc.add_root(path_input)
# doc.add_root(data_table)
# doc.add_root(slider_n_neighbours)
doc.add_root(full_row)
plot_col = column(name='plot_col')
plot_dropdown_col = column(distribution_dropdown,plot_col,name= 'plot_dropdown_col',width =300)
doc.add_root(plot_dropdown_col)
doc.add_root(tsne_figure)







