#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 16 13:09:18 2019

@author: SpaceMeerkat
"""

#=============================================================================#
#///   USEFUL INFORMATION   //////////////////////////////////////////////////#
#=============================================================================#

"""
To reconfigure the example script for loading in alternative datasets or saving
model results to alternative directories, change the following paths:
        
        DATA_PATH: dtype=str ; Directory of the data the user wishes to test
        MODEL_PATH: dtype=str ; Directory of the pre-trained model
        SAVE_PATH: dtype=str ; Directory to save model output results to
"""

#=============================================================================#
#///   CONFIGURABLE PATHS   //////////////////////////////////////////////////#
#=============================================================================#

DATA_PATH = '../Test_FITS_files/'
MODEL_PATH = 'CAE_Epoch_300.pt'
SAVE_PATH = '../Results/'
PCA_PATH = 'PCA_routine.pkl'
BOUNDARY = 2.960960960960961

#=============================================================================#
#///   END OF SCRIPT   ///////////////////////////////////////////////////////#
#=============================================================================#
