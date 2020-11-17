# MIR_Pipeline_Project


Multimedia retrieval pipeline for 3d models.





## Environment setup: ## 

Tested with Python version 3.7.
Only tested with Linux due to limited compatibility of the Faiss library with windows.

### With Anaconda (easy): ### 

Create an environment via the environment.yml file by calling the following:

```
conda env create --name envname --file=environments.yml
```

Activate environment:
```
conda activate envname
```

### Without Anaconda: ####

Install dependencies via requirements.txt by calling the following:

```
pip install -r requirements.txt
```

Since Faiss is not installable via pip it has to be compiled from source. This can be done
as detailed in the (official documentation)["https://github.com/facebookresearch/faiss/blob/master/INSTALL.md"]
Both the gpu and cpu versions of Faiss will work.


## Running the UI: ##
Make sure appropriate environment is activated.
CD into the src folder and call the following:

```
bokeh serve --show ui_viewer/
```

This should open as a page in your browser. If not you can use the localhost link that is shown in the terminal.



Using the UI:

1. Use the slider to set the number of query results.
2. Paste an absolute path to a .ply (non binary) or .off file into the text field located in the top left corner and press enter.
3. Wait for stats and graphs to appear.
4. Press the green "Visualize results" button to show the models on the canvas.

Which histograms are plotted can be selected via the orange dropdown.
Scroll down to see the t-SNE plot.
Canvas camera can be rotated by holding left mouse, panned by holding right nad zoomed using  the scroll button.

If something isn't working try reloading the page and if that fails restarting the server should do the trick. 


