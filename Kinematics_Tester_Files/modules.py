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
        """
        This class is used to import .fits images from a destination folder as a
        callable object for PyTorch's DataLoader function.
        
        Parameters
        ----------
        
        data.Dataset : str
                Path to folder containing .fits files that need to be loaded in
                
        Returns
        ----------
        
        dataset : object
                A dataset ready to be handled directly by PyTorch's DataLoader
                function. No specific returns are created by the class as it
                is not intended for use outside of the DataLoader handling.
        """
        def __init__(self,data_path):
                """
                This function initialises the parameters needed throughout the 
                functions inside FITSCubeDataset.
                
                Parameters
                ----------
                
                data_path : str
                        The directory path within which the .fits files are located for
                        loading in and processing
                        
                Returns
                ----------
                
                self.data_path : str
                        The directory of .fits files required for processing
                        
                self.IMG_EXTENSIONS : str
                        The extension for files required for processing
                        
                self._images : list
                        A list of all the files within a directory that end with the 
                        specified extension (in this case .fits)
                        
                """
                self.data_path = data_path
                self.IMG_EXTENSIONS = [".fits"]
                self._images = self.make_dataset()
        
        def is_image_file(self,filename):  
                """
                This function evaluates whether a file within a directory has a 
                specified extension. The only accepted extension for this class is the
                .fits extension.
                
                Parameters
                ----------
                
                filename : str 
                        The full directory path to a specified file
                        
                Returns
                ----------
                
                Result : bool
                        True or False verdict of the filename ending with the correct
                        extension
                """
                return any(filename.endswith(extension) for extension in self.IMG_EXTENSIONS)
        
        def transform_ALMA(self,_img):            
                """
                This function takes in a 2 dimensional velocity map as an array and 
                renormalises it to lie within the range {-1,1} and then clips any 
                extreme data using a percentile cut.
                
                Parameters
                ----------
                
                _img : np.ndarray 
                        The image the function will  rescale and clip
                        
                Returns
                ---------
                
                _img : ndarray
                        The rescaled and clipped velocity map
                """
                mins = np.nanpercentile(_img,2); _img -= mins
                maxs = np.nanpercentile(_img,98); _img /= maxs
                _img *= 2; _img -= 1; _img[_img != _img] = 0;
                _img[_img>1] = 0; _img[_img<-1] = 0
                return _img 
        
        def make_dataset(self): 
                """
                This function creates a list of all the .fits files from within the 
                directory specified in the parent class attributes.
                
                Parameters
                ----------
                
                None
                
                Returns
                ----------
                
                images : list
                        A list of all the files within a directory that end with the 
                        specified extension (in this case .fits)
                """
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
                """
                This function calls the numpy_transform_ALMA function on a .fits 
                file's data and converts the resultant np.ndarray into a torch 
                tensor ready for use with PyTorch's machine learning functionalities.
                
                Parameters
                ----------
                
                file_name : str
                        The path to the specified file complete with filename and 
                        extension. 
                        
                Returns
                ----------
                
                _training_data : torch.Tensor
                        A PyTorch tensor with 3 dimensions corresponding to the number
                        of channels, height, and width of the tensor. 
                """
                _file = fits.open(file_name)
                _data = _file[1].data.astype(float)  
                _data = torch.tensor(_data)
                _data = _data.unsqueeze(0).unsqueeze(0)
                _data = torch.nn.functional.interpolate(_data,size=64,mode='nearest')
                _data = _data.squeeze(0)
                _data = self.transform_ALMA(_data)
                _file.close()
                _training_data = (_data, os.path.basename(file_name)[:-5])
                return _training_data

        def __getitem__(self,index): 
                """
                This function processes an indexed .fits file ready for use with 
                PyTorch's machine learning functionalities. 
                
                Parameters
                ----------
                
                index : int
                        The index of a list of filenames from which the PyTorch 
                        DataLoader will call
                        
                Returns
                ----------
                
                _training_data : torch.Tensor
                        A PyTorch tensor with 3 dimensions corresponding to the number
                        of channels, height, and width of the tensor from an indexed
                        filename. 
                """
                _training_data = self.default_fits_loader(self._images[index])
                return _training_data 
        
        def __len__(self):            
                """
                This function returns the length of a list
                
                Parameters
                ----------
                
                None
                        PyTorch's DataLoader function requires this function but it is
                        not explicitly used by the user
                        
                Returns
                ----------
                
                len(self._images)
                        The total number of files within the dataset directory
                        containing the .fits extension
                """            
                return len(self._images)
        
#=============================================================================#
#/////////////////////////////////////////////////////////////////////////////#
#=============================================================================#