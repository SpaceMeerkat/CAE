# -*- coding: utf-8 -*-
"""
Created on Mon May 13 10:11:23 2019

@author: SpaceMeerkat
"""
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

import numpy as np
import torch
from astropy.io import fits
import os   
import torch.utils.data as data

#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#

IMG_EXTENSIONS = [".fits"]

class FITSCubeDataset(data.Dataset):
    def __init__(self,data_path):
        self.data_path = data_path
        self.IMG_EXTENSIONS = [".fits"]
        self._images = self.make_dataset()
        
    def is_image_file(self,filename):
        return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)
        
    def numpy_transform_ALMA(self,_img):
        mins = np.nanpercentile(_img,2); _img -= mins
        maxs = np.nanpercentile(_img,98); _img /= maxs
        _img *= 2; _img -= 1; _img[_img != _img] = 0;
        _img[_img>1] = 0; _img[_img<-1] = 0
        return _img 
        
    def make_dataset(self):
        directory = self.data_path
        images = []
        assert os.path.isdir(directory), '%s is not a valid directory' % directory
        for root, _, fnames in sorted(os.walk(directory)):
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        return images
        
    def default_fits_loader(self,file_name):
        _file = fits.open(file_name)
        _data = _file[1].data.astype(float)  
        _data = self.numpy_transform_ALMA(_data)
        _data = torch.tensor(_data)
        _data = _data.unsqueeze(0).unsqueeze(0)
        _data = torch.nn.functional.interpolate(_data,size=64)
        _data = _data.squeeze(0)
        _file.close()
        _training_data = (_data, os.path.basename(file_name)[:-5])
        return _training_data

    def __getitem__(self,index): 
        _training_data = self.default_fits_loader(self._images[index])
        return _training_data 
        
    def __len__(self):
        return len(self._images)
        
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#