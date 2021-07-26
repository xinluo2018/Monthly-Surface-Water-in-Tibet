import numpy as np
from utils.imgShow import imsShow
import matplotlib.pyplot as plt

def plot_dset_one(inputs, truth, pre, binary_out=True, weights=None):
    '''
    author: xin luo, date: 2021.3.24
    visualize input patches, prediction and the truth.
    input: 
        inputs: list, containinge input patches (np.array),
        truth: np.array. truth patch corresponding to the input
        binary: control the visualized image is a binary image or not
        weight: multi-scale weights corresponding to the input images
    '''
    num_input = len(inputs)
    if binary_out:
        pre = np.where(pre>0.5, 1, 0)
    if num_input == 1:
        patches_list = [inputs, truth, pre]
    else:
        patches_list = inputs+[truth, pre]
    patches_name = ['input_'+str(i) for i in range(num_input)] + ['truth','prediction']
    clip_list = [ 2 for i in range(num_input)] + [0, 0]
    col_bands_list = [(2,1,0) for i in range(num_input)] +[[0,0,0],[0,0,0]]
    imsShow(img_list=patches_list, img_name_list=patches_name, \
                    clip_list=clip_list, color_bands_list=col_bands_list)
    if weights:
        weights_name = ['weight_input_' + str(i) for i in range(num_input)]    
        format = [weight_name + ': {:f}' for weight_name in weights_name]
        format = ', '.join(format)
        print(format.format(*weights))
    plt.show()