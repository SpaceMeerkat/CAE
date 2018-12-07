# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import os
from tqdm import tqdm
from astropy.io import fits

IMG_EXTENSIONS = [".fits"]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_file_names(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images
    
def binner(array):
    array = np.round(np.round(array*200)/20)
    return array    
    
def binner2(array):
    array = np.round(array*10)/10
    return array 
    
def label_loader(file_name: str):
    file = fits.open(file_name)
    _label = file[0].header['KAPPA']
    file.close()    
    return _label
          
def weighting(directory):
    
    print('#'*73)
    
    images = get_file_names(directory)
    labels = []
    for i in tqdm(range(len(images))):
        labels.append([label_loader(images[i])]*1)
    labels = np.array(np.hstack(labels))
    target = binner(labels).astype(int)
    class_sample_count = np.array([(target == t).sum() for t in np.unique(np.sort(target))])
    weight = 1./class_sample_count
    final_weights = np.array([weight[t] for t in target])
    
    return final_weights
    
"""
    weight_array = np.zeros(10)

    for i in tqdm(range(len(labels))):
            
        if 0.0 < labels[i] <= 0.1:
            weight_array[0]+=1
        if 0.1 < labels[i] <= 0.2:
            weight_array[1]+=1
        if 0.2 < labels[i] <= 0.3:
            weight_array[2]+=1
        if 0.3 < labels[i] <= 0.4:
            weight_array[3]+=1
        if 0.4 < labels[i] <= 0.5:
            weight_array[4]+=1
        if 0.5 < labels[i] <= 0.6:
            weight_array[5]+=1
        if 0.6 < labels[i] <= 0.7:
            weight_array[6]+=1
        if 0.7 < labels[i] <= 0.8:
            weight_array[7]+=1
        if 0.8 < labels[i] <= 0.9:
            weight_array[8]+=1
        if 0.9 < labels[i] <= 1.0:
            weight_array[9]+=1

    weights_final = (len(labels)-weight_array)/len(labels)
    
    weight_array2 = []
    
    for i in tqdm(range(len(labels))):

        if 0.0 < labels[i] <= 0.1:
            weight_array2.append(weights_final[0])
        if 0.1 < labels[i] <= 0.2:
            weight_array2.append(weights_final[1])
        if 0.2 < labels[i] <= 0.3:
            weight_array2.append(weights_final[2])
        if 0.3 < labels[i] <= 0.4:
            weight_array2.append(weights_final[3])
        if 0.4 < labels[i] <= 0.5:
            weight_array2.append(weights_final[4])
        if 0.5 < labels[i] <= 0.6:
            weight_array2.append(weights_final[5])
        if 0.6 < labels[i] <= 0.7:
            weight_array2.append(weights_final[6])
        if 0.7 < labels[i] <= 0.8:
            weight_array2.append(weights_final[7])
        if 0.8 < labels[i] <= 0.9:
            weight_array2.append(weights_final[8])
        if 0.9 < labels[i] <= 1.0:
            weight_array2.append(weights_final[9])
        
    return np.ones(len(weight_array2))
"""            
            