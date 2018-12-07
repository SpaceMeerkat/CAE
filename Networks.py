# -*- coding: utf-8 -*-
"""
Created on Fri Sep 14 10:56:03 2018

@author: c1307135
"""

import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

class CategoricalNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            
            torch.nn.Conv2d(1,64,5,padding=2), 
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
		 torch.nn.Dropout(0.5),
            torch.nn.Linear(256*16*16,256), # Fully connected layer 
            torch.nn.ReLU(),            
            torch.nn.Linear(256,5))
        
    def forward(self,x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]),-1))
        output= torch.nn.functional.log_softmax(output,dim=1)
        return output
        
class RegressionalNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            
            torch.nn.Conv2d(1,64,3,padding=1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128,256,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
             
        self.classifier = torch.nn.Sequential(
            #torch.nn.Dropout(0.2),
            torch.nn.Linear(256*16*16,256),
            torch.nn.ReLU(),    
            torch.nn.Linear(256,256),
            torch.nn.ReLU(),       
            torch.nn.Linear(256,1))
        
    def forward(self,x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]),-1))
        #output= torch.nn.functional.log_softmax(output,dim=1)
        return output

class RegressionalNet2(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            
            torch.nn.Conv2d(1,32,6,padding=3), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,5,padding=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128,128,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
             
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(128*8*8,128*8*8),
            torch.nn.ReLU(),  
            torch.nn.Dropout(0.1),       
            torch.nn.Linear(128*8*8,128*8*8),
            torch.nn.ReLU(),
            torch.nn.Linear(128*8*8,1))
        
    def forward(self,x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]),-1))
        #output= torch.nn.functional.log_softmax(output,dim=1)
        return output
        
class BranchedNet(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            
            torch.nn.Conv2d(1,64,3,padding=1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128,256,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
            
        self.feature_extractor2 = torch.nn.Sequential(
            
            torch.nn.Conv2d(1,64,3,padding=1), 
            torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128,256,3,padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
             
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear((16*16*256)*2,264),
            torch.nn.ReLU(),      
            torch.nn.Linear(264,264),
            torch.nn.ReLU(),
            torch.nn.Linear(264,1))
        
    def forward(self,x,y):
        features = self.feature_extractor(x)
        features = features.view(int(x.size()[0]),-1)
        features2 = self.feature_extractor(y)
        features2 = features2.view(int(y.size()[0]),-1)
        grouped = torch.cat((features,features2),dim=1)
        output = self.classifier(grouped)

        return output
        
class RegressionNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(1, 64, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, 5, padding=2),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 5, padding=2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 512, 3, padding=1),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(512),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(512, 256, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(256),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 128, 3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1),
            torch.nn.ReLU(),
            #torch.nn.BatchNorm2d(128),
            torch.nn.MaxPool2d(2))
        self.classifier = torch.nn.Sequential(
            #torch.nn.Dropout(0.75),
            torch.nn.Linear(128 * 2 * 2, 256),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.75),
            torch.nn.Linear(256, 1))

    def forward(self, x):
        features = self.feature_extractor(x)
        output = self.classifier(features.view(int(x.size()[0]), -1))
        #output= torch.nn.functional.log_softmax(output,dim=1) # Give results using softmax
        return output
        
        
def plot_accuracy(accuracies, val_acc, top_3, epochs, val_epochs,IMG_PATH, filename):
    fig = plt.figure()
    ax = fig.gca()
    ax.set_xlim(0, max(epochs))
    ax.set_ylim(0, 105)
    plt.plot(epochs, accuracies,'b',label='Training Accuracy',zorder=1)
    plt.plot(val_epochs, val_acc,'purple',label='Validation Accuracy',zorder=0)
    plt.plot(val_epochs, top_3,'g',marker='^',linestyle='None',label='Top-2 Validation Accuracy',zorder=0)
    plt.ylabel('Accuracy')
    plt.xlabel('Training Epoch')
    plt.legend(loc='best',fontsize='small')
    fig.savefig(IMG_PATH+filename, bbox_inches='tight')
    plt.close()
    
def plot_accuracy_regression(accuracies, val_acc, running_val_acc, epochs, val_epochs,IMG_PATH, filename):
    fig = plt.figure(figsize=(20,10))
    ax = fig.gca()
    ax.set_xlim(0, max(epochs))
    plt.plot(epochs, accuracies,'b',label='Training Loss',zorder=1)
    plt.plot(val_epochs,running_val_acc,'purple',label='Validation Accuracy',zorder=0)
    plt.ylabel('MSE Loss')
    plt.xlabel("Training Epoch'")
    plt.legend(loc='best',fontsize='small')
    fig.savefig(IMG_PATH+'_Training_'+filename, bbox_inches='tight')
    fig = plt.figure(figsize=(20,10))
    for i in range(len(val_acc[0,:])):
        plt.plot(val_epochs, val_acc[:,i],label='Validation Loss group '+str(i/10.),zorder=0)
    plt.ylabel('MSE Loss')
    plt.xlabel("Epoch'")
    plt.legend(loc='best',fontsize='small')
    fig.savefig(IMG_PATH+filename, bbox_inches='tight')
    plt.close()
    
    
def PIL_transform(_img):
    if _img.shape[0] == 1:
        _img[_img != _img] = 0.
        _img -= _img.min()
        _img *= 255./_img.max()
        _img = _img.astype(np.uint8)
    else:
        for i in range(_img.shape[0]):
            _img[i,:,:][_img[i,:,:] != _img[i,:,:]] = 0.
            _img[i,:,:] -= _img[i,:,:].min()
            _img[i,:,:] *= 255./_img[i,:,:].max()
            _img[i,:,:] = _img[i,:,:].astype(np.uint8)
    return _img 
    
def numpy_transform(_img):
    for i in range(_img.shape[0]):
        _img[i] -= _img[i].min()
        _img[i] *= 255./_img[i].max()
        _img[i] = _img[i].astype(np.uint8)
    return _img 
    
    
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()
       
'''        
def weight_init(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
'''            
def labeller(label):
    if label <= 5:
        _label = 0
    if label == 6:
        _label = 1
    if label == 7:
        _label = 2
    if label == 8:
        _label = 3
    if label == 9:
        _label = 4
    return _label
    
def topk_accuracy(output, target, topk=(3)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
    
def binner(array):
    decs = np.floor(array*10.)/10.
    bins = np.arange(0,1.0,0.1)
    n = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(bins)):
        n[i] = np.where(decs==bins[i]) 
        if len(np.where(bins[i]==decs)[0]) == 0:
            n[i] = np.nan
    return n
    
def bin_errors(wheres,loss_array):
    n = [[],[],[],[],[],[],[],[],[],[]]
    for i in range(len(n)):
        if np.isnan(wheres[i][0]) == True:
            n[i] = np.array(np.nan)
        else:
            n[i] = np.mean(loss_array[wheres[i][0]])
    return n

def no_nans(_img):
    _img[_img != _img] = 0 
    return _img