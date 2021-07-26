import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import make_interp_spline

def smooth(x, window=31):
    _head = np.full(shape=int((window-1)/2), fill_value=x[0])
    _tail = np.full(shape=int((window-1)/2), fill_value=x[-1])
    s = np.r_[_head, x, _tail]
    w = np.ones(window,'d')
    y = np.convolve(w/w.sum(), s, mode='valid')
    return y

def metrics_plot(metrics, keys, colors, axis_name=['Iterations', 'Accuracy'], \
                axis_range=None, legend_name=None, legend_pos=None, y_ticks=None, smooth_window = 31):
    '''des: plot accuracy metrics
       input metrics is pandas.dataframe
    '''
    if axis_range is None:
        axis_range = [[0,metrics.shape[0]], [0, 1]]
    if legend_name is None:
        legend_name = keys
    if y_ticks is None:
        y_ticks = [axis_range[1][0]+(i/5)*(axis_range[1][1]-axis_range[1][0]) 
                                                        for i in range(5)]
    plt.figure(figsize=(6, 4))
    ax = plt.subplot(111)
    ## insert epoches column
    if 'epoch' not in metrics.columns:
        metrics.insert(0, 'epoch', np.arange(metrics.shape[0]))
    for i, key in enumerate(keys):
        ## ----- smooth ---- 
        metrics[key+'_smooth'] = smooth(metrics[key].to_numpy(), window=smooth_window) 
        ## ----- interpolate -----
        x_y_spline = make_interp_spline(metrics['epoch'], metrics[key+'_smooth'])
        x_new_d = np.linspace(metrics['epoch'].min(), metrics['epoch'].max(), 20)
        y_new_d = x_y_spline(x_new_d)
        ## ------ plot -------
        plt.plot(metrics['epoch'], \
                    metrics[key], \
                    color=colors[i], linestyle='dotted', \
                    linewidth = 0.7)
        plt.plot(x_new_d, y_new_d, color=colors[i],\
                    linewidth = 1.5, marker='o', \
                    markerfacecolor=colors[i], markersize=4, \
                    label= legend_name[i])

    ## ----- figure setting ----- 
    if legend_pos is None:
        plt.legend(prop = {'size':10}, loc= 'best')
    else:
        plt.legend(prop = {'size':10}, loc= 'best', bbox_to_anchor=legend_pos)    
    ax=plt.gca()
    y_major_locator=MultipleLocator(0.1)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.tick_params(labelsize=12)
    plt.xlim(axis_range[0])
    plt.ylim(axis_range[1])
    ax.set_yticks(y_ticks)
    plt.xlabel(axis_name[0], fontsize=14) 
    plt.ylabel(axis_name[1], fontsize=14)
