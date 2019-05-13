# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:31:52 2019

@author: SpaceMeerkat
"""
#=============================================================================#
#///   Load the required packages   //////////////////////////////////////////#
#=============================================================================#

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

from Testing_modules import FITSCubeDataset

#=============================================================================#
#///   Load the data and test   //////////////////////////////////////////////#
#=============================================================================#

""" Define the paths to your velocity maps and the CAE model path """

_data_path = '/home/user/Documents/data/'
_model_path = '/home/user/Documents/model/'

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

""" Load the CAE model and set it to evaluation mode """

model = torch.load(_model_path+'CAE.pt').cpu()
model.train(False)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

""" Load the data using a given batch size (default=64) and given number of 
workers (default=16) """

test_loader = DataLoader(dataset=FITSCubeDataset(_data_path),
                          batch_size=64,num_workers=16,shuffle=True)      
                          
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#
                          
""" Pass the loaded images through the CAE model and store the results """
                          
all_names = []
all_features = []

for idx, (batch, names) in tqdm(enumerate(test_loader)):
    features,ind1,s1,ind2,s2 = model.encoder(batch.float())
    all_names.append(names)
    all_features.append(features.detach().numpy())

all_features = np.hstack(all_features)
all_names = np.hstack(all_names)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

""" Optional: Store the results as a DataFrame and pickle """

df_data = np.vstack([all_names,all_features.T]).T

results = pd.DataFrame(df_data)
results.columns = ['Name','L1','L2','L3']
results.set_index(keys='Name',drop=True,inplace=True)

_save_path = '/home/user/Documents/results/'
results.to_pickle(_save_path+'CAE_results.pkl')

#=============================================================================#
#///   End of script   ///////////////////////////////////////////////////////#
#=============================================================================#