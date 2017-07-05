from plotly.graph_objs import Scatter, Figure, Layout, Data
from plotly import tools
import plotly
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import numpy as np
def plotRandomOffline(x,y):
    print("PRINT X", x)
    print("PRINT Y", y)
    plot([Scatter(x=x, y=y)])

def plotSeries(y):
    plot([Scatter(x=np.arange(len(y)), y=y)])

def plotMultiSeries(array_y):
    fig = tools.make_subplots(rows=len(array_y), cols=1)
    i = 1
    for y in array_y:
        fig.append_trace(Scatter(x=np.arange(len(y)), y=y), i, 1)
        i = i+1
    plot(fig)
