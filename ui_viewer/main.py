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




print(bokeh_paths.ROOT_DIR)

print(bokeh_paths.serverdir())

plot = figure(plot_width = 400,tools = 'pan,box_zoom')
plot.circle([1,2,3,4,5],[8,6,3,2,1],name='test_plot')



#curdoc().add_root(plot)
path = 'C:/Users/samme/Google_Drive/Code_library/MIR_Pipeline_Project/ui_viewer/templates/Lucy100k.ply'
path = 'ui_viewer/static/Lucy100k.ply'
path = 'ui_viewer/static/m1405.ply'
curdoc().template_variables['ply_path'] = path



