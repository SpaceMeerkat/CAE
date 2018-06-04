# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
def weighting(labels):
    
    weight_array = np.zeros(10)

    for i in range(len(labels)):
        if labels[i] == 0:
            weight_array[0]+=1    
        if labels[i] == 1:
            weight_array[1]+=1
        if labels[i] == 2:
            weight_array[2]+=1
        if labels[i] == 3:
            weight_array[3]+=1
        if labels[i] == 4:
            weight_array[4]+=1
        if labels[i] == 5:
            weight_array[5]+=1
        if labels[i] == 6:
            weight_array[6]+=1
        if labels[i] == 7:
            weight_array[7]+=1
        if labels[i] == 8:
            weight_array[8]+=1
        if labels[i] == 9:
            weight_array[9]+=1
            
    weights_final = (len(labels)-weight_array)/len(labels)
    
    weight_array2 = []
    
    for i in range(len(labels)):
        if labels[i] == 0:
            weight_array2.append(weights_final[0])
        if labels[i] == 1:
            weight_array2.append(weights_final[1])
        if labels[i] == 2:
            weight_array2.append(weights_final[2])
        if labels[i] == 3:
            weight_array2.append(weights_final[3])
        if labels[i] == 4:
            weight_array2.append(weights_final[4])
        if labels[i] == 5:
            weight_array2.append(weights_final[5])
        if labels[i] == 6:
            weight_array2.append(weights_final[6])
        if labels[i] == 7:
            weight_array2.append(weights_final[7])
        if labels[i] == 8:
            weight_array2.append(weights_final[8])
        if labels[i] == 9:
            weight_array2.append(weights_final[9])
            
    return weight_array2
            
            