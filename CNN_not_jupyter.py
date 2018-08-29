#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 12:19:02 2018

@author: jamesdawson
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
import numpy as np
from sklearn.metrics import accuracy_score
from torchvision import transforms
from astropy.io import fits 
from skimage.transform import resize
import time
import matplotlib
from matplotlib import pyplot as plt
from weighting import weighting

plt.ioff()
plt.close('all')

torch.cuda.benchmark=True

DATA_PATH ='/home/corona/c1307135/TRAINING/Quick_Test/Cubes_mid_28/'
VAL_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/EXAMPLES/'
IMG_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/OUTPUT/'

# In[]:

class CategoricalNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            
            torch.nn.Conv2d(1,64,5,padding=2), # 1 input, 32 out, filter size = 5x5, 2 block outer padding
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,5,padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128,256,5,padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2))
 
        self.classifier = torch.nn.Sequential(
		    torch.nn.Dropout(0.25),
            torch.nn.Linear(256*16*16,256), # Fully connected layer 
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256,10))
        
    def forward(self,x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]),-1))
        output= torch.nn.functional.log_softmax(output,dim=1)
        return output

# In[]:
        
class RegressionNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2))
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.25),
            torch.nn.Linear(128 * 2 * 2, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(256, 10))

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]), -1))
        output= torch.nn.functional.log_softmax(output,dim=1) # Give results using softmax
        return output

# In[]:
        
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
        
# In[]:
        
def PIL_transform(_img):
    _img[_img != _img] = 0
    _img -= _img.min()
    _img *= 255./_img.max()
    _img = _img.astype(np.uint8)
    return _img 

# In[]:
    
import torch.utils.data as data

IMG_EXTENSIONS = [
    ".fits"
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images


def default_fits_loader(file_name: str, img_size: tuple, slice_index):
    file = fits.open(file_name)
    _data = file[1].data
    _data = resize(_data[slice_index], img_size)
    _label = file[0].header['LABEL']
    file.close()

    if len(_data.shape) < 3:
        _data = _data.reshape((*_data.shape, 1))
    
    return _data, _label


class FITSCubeDataset(data.Dataset):
    def __init__(self, data_path, cube_length, transforms, img_size):
        self.data_path = data_path
        self.transforms = transforms
        self.img_size = img_size
        self.cube_length = cube_length
        self.img_files = make_dataset(data_path)

    def __getitem__(self, index):
        cube_index = index // self.cube_length
        slice_index = index % self.cube_length
        _img, _label = default_fits_loader(self.img_files[cube_index], self.img_size, slice_index)
        _img[_img != _img] = 0
        _img = PIL_transform(_img)
        if self.transforms is not None:
            _data = (self.transforms(_img), _label)
        #else:
        #    _data = (_img, _label)
            
        return _data

    def __len__(self):
        return len(self.img_files)*self.cube_length

# In[]:
        
def plot_accuracy(accuracies, val_acc, epochs, val_epochs, filename):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 105)
    plt.plot(epochs, accuracies,'b',label='Training Accuracy',zorder=1)
    plt.plot(val_epochs, val_acc,'purple',marker='^',linestyle='None',label='Validation Accuracy',zorder=0)
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend(loc='best',fontsize='small')
    fig.savefig(IMG_PATH+filename, bbox_inches='tight')
    plt.close()
    
# In[]:
    
def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    decay = initial_lr / num_epochs
    lr = initial_lr - decay*epoch
    print("Set LR to %f" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
# In[]:
        
def train(model: torch.nn.Module, 
          transforms, 
          data_path= DATA_PATH, 
          val_path= VAL_PATH, 
          train_batch_size = 339839,
          val_batch_size = 52479,
          num_epochs=50, 
          batch_size_const=64, 
          verbose=True,
          cube_length=640, img_size=(64, 64), 
          loss=torch.nn.CrossEntropyLoss(), 
          lr_schedule=True, initial_lr=1e-3, suffix=""):

    data_path = os.path.abspath(data_path)
    val_path = os.path.abspath(val_path)
	
    model = model.train()
    device = torch.device("cuda")
    model = model.to(device).to(torch.float)
    start = time.time()
    
    """ LOADING IN THE TRAINING DATA FOR WEIGHT CALCULATIONS """
    
    print('Creating sampling weight array')
    train_loader = DataLoader(FITSCubeDataset(data_path, cube_length, transforms, img_size), 
                              batch_size=train_batch_size, shuffle=False)
    dataiter = iter(train_loader)
    dummy_labels = []
    for idx, (batch, target) in enumerate(tqdm(train_loader)):
        dummy_labels.append(np.array(target.numpy()))
    dummy_labels = np.hstack(dummy_labels)
    print(len(dummy_labels))
    print('Number of labels=',len(set(dummy_labels)))
    weights = weighting(dummy_labels)
    print('Number of different samplers = ', len(set(dummy_labels)))

    sampler = WeightedRandomSampler(weights, len(weights))
    end = time.time()
    print('Weights Created in %.2gs'%(end-start))
    
    """ LOADING IN THE VALIDATION DATA FOR WEIGHT CALCULATIONS """
    #batch size was 10*640?
    start = time.time()
    val_loader = DataLoader(FITSCubeDataset(val_path, cube_length, transforms, img_size), 
                            batch_size=val_batch_size, shuffle=False)
    dataiter = iter(val_loader)
    dummy_val_labels = []
    for idx, (batch, target) in enumerate(tqdm(val_loader)):
        dummy_val_labels.append(np.array(target.numpy()))
    dummy_val_labels = np.hstack(dummy_val_labels)
    print(len(dummy_val_labels))
    print('Number of labels=',len(set(dummy_val_labels)))
    val_weights = weighting(dummy_val_labels)
    val_sampler = WeightedRandomSampler(val_weights, len(val_weights))
    end = time.time()
    print('Validation weights Created in %.2gs'%(end-start))
    
    """ LOADING IN THE REAL TRAINING AND VALIDATION DATASETS """
    
    loader = DataLoader(FITSCubeDataset(data_path, cube_length, transforms, img_size), 
                        batch_size=batch_size_const, shuffle=False, sampler=sampler)
                        
    validation_loader = DataLoader(FITSCubeDataset(val_path, cube_length, transforms, img_size), 
                                   batch_size=batch_size_const, shuffle=False, sampler=val_sampler)   
    
    optim = torch.optim.Adam(model.parameters(), initial_lr)
	
    accuracies, val_accuracies, epochs, val_epochs = [0], [0], [0], [0]
	
    for i in range(num_epochs):
        print("Epoch %d of %d" % (i+1, num_epochs))
        _accuracies,_val_accuracies = [],[]
        
        model.train(True) 
        for idx, (batch, target) in enumerate(tqdm(loader)):
            batch = batch.to(device).to(torch.float)
            if isinstance(loss, torch.nn.CrossEntropyLoss):
                target = target.to(device).to(torch.long)
            else:
                target = target.to(device).to(torch.float)
            pred = model(batch)

            loss_value = loss(pred, target)

            optim.zero_grad()
            loss_value.backward()
            optim.step()

            pred_npy = pred.detach().cpu().numpy()
            target_npy = target.detach().cpu().numpy()

            if isinstance(loss, torch.nn.CrossEntropyLoss):
                pred_npy = np.argmax(pred_npy, axis=1) 
                
            ###Change the error metric here###

            pred_int = np.round(pred_npy).astype(np.uint8).reshape(-1)
            target_npy = target_npy.astype(np.uint8).reshape(-1)

            _accuracies.append(accuracy_score(target_npy, pred_int)*100)
            
        epochs.append(i+1)
        
        

        mean_accuracy = sum(_accuracies)/len(_accuracies)
        accuracies.append(mean_accuracy)

        print("Mean accuracy: %f" % mean_accuracy)
        
        model.train(False)
        
        if i % 5 == 0:
            
            val_epochs.append(i+1)

            for idx, (batch, target) in enumerate(tqdm(validation_loader)):
                batch = batch.to(device).to(torch.float)
                if isinstance(loss, torch.nn.CrossEntropyLoss):
                    target = target.to(device).to(torch.long)
                else:
                    target = target.to(device).to(torch.float)
                pred = model(batch)
    
                loss_value = loss(pred, target)
    
                pred_npy = pred.detach().cpu().numpy()
                target_npy = target.detach().cpu().numpy()
    
                if isinstance(loss, torch.nn.CrossEntropyLoss):
                    pred_npy = np.argmax(pred_npy, axis=1) 
                    
                ###Change the error metric here###
    
                pred_int = np.round(pred_npy).astype(np.uint8).reshape(-1)
                target_npy = target_npy.astype(np.uint8).reshape(-1)
    
                _val_accuracies.append(accuracy_score(target_npy, pred_int)*100)
    
            mean_val_accuracy = sum(_val_accuracies)/len(_val_accuracies)
            val_accuracies.append(mean_val_accuracy)
            print("Mean Validation accuracy: %f" % mean_val_accuracy)
            
        if lr_schedule:
            plot_accuracy(accuracies,val_accuracies, epochs, val_epochs, "Validation_accuracy_scheduler%s_25Mpc_RegressionNet.png" % suffix)
        else:
            plot_accuracy(accuracies,val_accuracies, epochs, val_epochs, "Validation_accuracy_no_scheduler%s_25Mpc_RegressionNet.png" % suffix)
        

        model.eval()
        
        if i % 10 == 0:
            torch.save(model, IMG_PATH+'RegressionNet_25Mpc.pt')
        
# In[]:
        
if __name__ == '__main__':
    print("Creating Model and Initializing weights")
	
#    for model_class, loss_fn, suffix in zip([CategoricalNet, RegressionNet], [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()], ["_categorical", "_regression"]):
#        for schedule in [True, False]:
            
    model_class, loss_fn, suffix = RegressionNet, torch.nn.CrossEntropyLoss(), "_categorical"
    schedule = True
    
    model = model_class()
    model.apply(weight_init)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

start = time.time()
train(model, transform,  num_epochs=21, batch_size_const=64, lr_schedule=schedule, loss=loss_fn, suffix=suffix)
end = time.time()
print('TRAIN TIME:')
print('%.2gs'%(end-start))

# In[]:

torch.save(model, IMG_PATH+'RegressionNet_25Mpc.pt')
