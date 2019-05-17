# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:31:52 2019

@author: SpaceMeerkat
"""
#=============================================================================#
#///   Load the required packages   //////////////////////////////////////////#
#=============================================================================#

import os

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import *
from modules import FITSCubeDataset

#=============================================================================#
#///   Load the data and begin testing   /////////////////////////////////////#
#=============================================================================#

""" Load the CAE model and set it to evaluation mode on a cpu device """

model = torch.load(MODEL_PATH,map_location='cpu').cpu()
model.train(False)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

""" Load the data using a given batch size (default=64) and given number of 
workers (default=16) """

test_loader = DataLoader(dataset=FITSCubeDataset(DATA_PATH),
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

""" Load the PCA routine and transform the 3D latent data for making 
    predictions of circularity """

with open(PCA_PATH, 'rb') as file:  
    PCA_model = pickle.load(file)
    
pca_test_data = PCA_model.transform(all_features)

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

""" Get a circularity prediction from the 3D latent positions """

latent_x = np.sqrt((pca_test_data[:,0]**2)+(pca_test_data[:,1]**2))
latent_y = np.abs(pca_test_data[:,2])
classifications = np.ones(len(latent_x))
zero_indices = np.where(latent_x<BOUNDARY)[0]
classifications[zero_indices] = 0

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

""" Optional: Store the results as a DataFrame and pickle """

df_data = np.vstack([all_names,all_features.T,classifications]).T
results = pd.DataFrame(df_data)
results.columns = ['Name','L1','L2','L3','Circularity']
results.set_index(keys='Name',drop=True,inplace=True)
results.to_pickle(os.path.join(SAVE_PATH,'CAE_results.pkl'))

#=============================================================================#
#///   End of script   ///////////////////////////////////////////////////////#
#=============================================================================#