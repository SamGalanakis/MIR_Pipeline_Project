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
import pandas as pd
from bokeh.plotting import figure,curdoc,show,save
import colorcet as cc

def make_tsne(tsne_csv_path):
    df = pd.read_csv(tsne_csv_path)
    df['x_data']
   
    unique_classes = sorted(list(set(df['classification'])))
    classification_indexes = [unique_classes.index(x) for x in df['classification']]
    
    
    colors =cc.b_glasbey_bw[0:len(unique_classes)]
    draw_colors = [colors[classification_indexes[x]] for x in range(df.shape[0])]
    

    #colors_rgb = mpl.colors.hsv_to_rgb(colors)
    tooltips =  """
    <div>
        <div>
            <img
                src="@thumbnails" height="42" width="42"
                style="float: left; margin: 0px 15px 15px 0px;"
                border="2"
            ></img>
        </div>
        <div>
            <span style="font-size: 17px; font-weight: bold;">@name</span>
            <span style="font-size: 15px; color: #966;">[@classification]</span>
        </div>
    
        <div>
            
            <span style="font-size: 10px; color: #696;">($x, $y)</span>
        </div>
    </div>
"""
  
    hover = HoverTool(tooltips=tooltips)




    colors = [tuple(x) for x in colors]


   
    thumbnails = [f'ui_viewer/static/thumbnails/{name}_thumb.jpg' for name in df['name']]
    
    data = dict(x=df['x_data'],y=df['y_data'],thumbnails=thumbnails,
    classification=df['classification'],name=df['name'],colors=draw_colors)
    source = ColumnDataSource(data)
    p = figure(plot_width=1000, plot_height=1000, tools=[hover], title="TSNE",
    name='tsne_figure')
    p.circle('x', 'y', size=10, source=source,
         color='colors',alpha=255)
    
    show(p)
    return p


if __name__ == '__main__':



    tsne_path = os.path.join(os.path.dirname(src_dir),'processed_data/tsne_data.csv')
    make_tsne(tsne_path)