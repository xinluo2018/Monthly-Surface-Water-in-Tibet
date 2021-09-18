import numpy as np


## max and min value for the s1 ascending/descending images
s1_min = [-57.78, -70.37, -58.98, -68.47]  # as-vv, as-vh, des-vv, des-vh
s1_max = [25.98, 10.23, 29.28, 17.60]

def get_s1pair_nor(s1_as, s1_des):
    '''read in and normalize the s1 acending and descending pair image'''
    s1_img = np.concatenate((s1_as, s1_des), axis=2)
    s1_img_nor = s1_img.copy()
    ## normalization.
    for j in range(s1_img.shape[-1]):
        s1_img_nor[:,:,j] = (s1_img[:,:,j] - s1_min[j])/(s1_max[j]-s1_min[j]+0.0001)
    image = np.clip(s1_img_nor, 0., 1.) 
    image[np.isnan(image)]=0         # remove nan value
    return image


