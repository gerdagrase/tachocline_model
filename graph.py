# code based on the examples from:
#https://www.kite.com/python/answers/how-to-plot-data-from-a-csv-file-in-python and https://www.kite.com/python/answers/how-to-use-a-colormap-to-set-the-color-of-lines-in-a-matplotlib-line-graph-in-python 
# both last accessed 19th March 2021

# import the required modules for plotting and shading the graph 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
import argparse
import warnings

parser = argparse.ArgumentParser(description="Graph producing code")
parser.add_argument("--graph",    default='2D',               type=str, help="choice of model: 2D, 3D, or time; select time for time dependent graphs")
parser.add_argument("--var",      action="extend", nargs="+", type=str, help="variables for the x and y (and z) axes; 0 - time, 1-6 - x_i, 7-12 - y_i, 13-18 - z_i; enter as 'value value value'")
parser.add_argument("--filename", action="extend", nargs="+", type=str, help="selection of plotted files; enter as 'value value value' for multiple files")
parser.add_argument("--label",    default='Figure_graph',     type=str, help="file label")
parser.add_argument("--save",     action='store_true',                  help="save figure instead of displaying?")
parser.add_argument("--n",        default=1,                  type=int, help="plot every n-th point")
parser.add_argument("--k",        default=0,                  type=int, help="plot the last k points")

args = parser.parse_args()

graph = args.graph
var = args.var
filename = args.filename
label = args.label
save = args.save
n = args.n
k = args.k

# array from which to pull axes labels
labels = [r'$t$', r'$x_1$', r'$x_2$', r'$x_3$', r'$x_4$', r'$x_5$', r'$x_6$', r'$y_1$', r'$y_2$', r'$y_3$', r'$y_4$', r'$y_5$', r'$y_6$', r'$z_1$', r'$z_2$', r'$z_3$', r'$z_4$', r'$z_5$', r'$z_6$'] 

# define initial figure name
figname = label+'_'+str(graph)+'_vars_'+str(var[0])+'_'+str(var[1])

if graph == '2D':
    # read the .csv file and save the necessary variables
    plots = np.genfromtxt(str(filename[0]), delimiter=",", skip_header=1, usecols=(int(var[0]), int(var[1])))
    x = plots[:,0]
    y = plots[:,1]
    
    # which number each point is, to colour it according to colourmap
    colorindex = range(int(len(x[-k:])/n))

    # create a scatter plot in the colourmap plasma
    plt.scatter(x[-k::n], y[-k::n], c = colorindex[:], cmap = "plasma", marker = ".", s = 0.5)
    
    plt.xlabel(str(labels[int(var[0])]))
    plt.ylabel(str(labels[int(var[1])]))
    
elif graph == '3D':
    # read the .csv file and save the necessary variables  
    plots = np.genfromtxt(str(filename[0]), delimiter=",", skip_header=1, usecols=(int(var[0]), int(var[1]), int(var[2])))
    x = plots[:,0]
    y = plots[:,1]
    z = plots[:,2]
    
    # which number each point is, to colour it according to colourmap
    colorindex = range(int(len(x[-k:])/n)) 

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')

    # create a scatter plot in the colourmap plasma
    axes.scatter(x[-k::n], y[-k::n], z[-k::n], c = colorindex[:], cmap = "plasma", marker = ".", s = 0.4)
    
    axes.set_xlabel(str(labels[int(var[0])]))
    axes.set_ylabel(str(labels[int(var[1])]))
    axes.set_zlabel(str(labels[int(var[2])]))
    
    figname = figname+'_'+str(var[2])
    
elif graph == 'time':    
    if int(var[0])!=0 and int(var[1])!=0:
        warnings.warn("No time variable selected; graph may be cluttered and difficult to view")
    
    # define colours
    interval = np.linspace(0, 1, len(filename))
    colors = [cm.plasma(x) for x in interval]
    
    for i in range(len(filename)):
        # read the specified columns in a file
        plots = np.genfromtxt(str(filename[i]), delimiter=",", skip_header=1, usecols=(int(var[0]), int(var[1])))
        x = plots[:,0]
        y = plots[:,1]
        
        # create a plot
        plt.plot(x[-k::n], y[-k::n] , c = colors[i], linewidth=0.5)
    
    plt.xlabel(str(labels[int(var[0])]))
    plt.ylabel(str(labels[int(var[1])]))
    
if save:
    # layout to fix cropping issues with axes
    plt.tight_layout()
    plt.savefig(figname, dpi=200)
else:
    plt.show()
