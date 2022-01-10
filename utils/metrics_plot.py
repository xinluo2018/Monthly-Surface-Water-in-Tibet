## author: xin luo
## create: 2021.10.26; modify: 2021.12.7
## des: function for ploting the obtained metrics, e.g., loss, oa.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from scipy.interpolate import make_interp_spline

def smooth(y, window=31, num_sam = None):
    _head = np.full(shape=int((window-1)/2), fill_value=y[0])
    _tail = np.full(shape=int((window-1)/2), fill_value=y[-1])
    s = np.r_[_head, y, _tail]
    w = np.ones(window,'d')
    y = np.convolve(w/w.sum(), s, mode='valid')
    x = np.arange(y.shape[0])
    if num_sam is not None:
        i_iter = np.arange(y.shape[0])
        x_y_spline = make_interp_spline(i_iter, y)
        x = np.linspace(i_iter.min(), i_iter.max(), num_sam)
        y = x_y_spline(x)
    return x, y

def csv_merge(csv_files, i_csv = None, i_row=None, sam=None):
    '''
    des: merge the csv file along the column, the metrics should be the same size.
    args:
      metric: .csv file path or pandas.dataframe data. (the data are with column name).
      i_csv: str, the name of the new created column corresponding csv file id.
      i_row: str, the name of the new created column corresponding row id of the csv file.
    return:
      metrics_model: the merged metrics
    '''
    for i, metric in enumerate(csv_files):
        if isinstance(metric, str):
          metric = pd.read_csv(metric) 
        metric_proc = metric.copy()
        if sam:
          metric_proc = metric_proc[::sam]     # down-sampling
        if i_row is not None:
          metric_proc[i_row] = metric_proc.index+1
        if i_csv is not None:
          metric_proc[i_csv] = np.ones_like(metric_proc.index) + i
        if i == 0:
          metrics_merge = metric_proc   # initial metrics_model
        else:
          metrics_merge = pd.concat([metrics_merge, metric_proc], axis = 0)
    metrics_merge.index = range(0,len(metrics_merge))   # # 
    return metrics_merge


