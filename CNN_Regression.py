# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 09:16:34 2018

@author: c1307135
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
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from weighting import weighting, is_image_file, get_file_names, label_loader
from Networks import RegressionalNet, BranchedNet, plot_accuracy_regression,PIL_transform,weight_init,binner,bin_errors, numpy_transform,no_nans
from PIL import Image

#import torchvision.models as models
#
#resnet = models.resnet50(pretrained=True)
#resnet.conv1 = torch.nn.Conv2d(1,64, kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
#resnet.fc = torch.nn.Linear(2048,1)
#ct = 0
#for child in resnet.children():
#    ct+=1
#if ct < 7:
#    for param in child.parameters():
#        param.requires_grad = False


#torch.cuda.empty_cache()

plt.ioff()
plt.close('all')

torch.cuda.benchmark=True
torch.cuda.fastest =True
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(0))

#DATA_PATH ='/home/anubis/c1307135/Grouped_sets/new_grouped/TRAIN/'
DATA_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/test_images/'
VAL_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/test_images/'
#VAL_PATH = '/home/anubis/c1307135/Grouped_sets/new_grouped/TEST/'
OUTPUT_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/test_folder/'
IMG_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/OUTPUT/Overfitting_Test/031/' 
MODEL_PATH = '/home/corona/c1307135/TRAINING/Quick_Test/OUTPUT/Overfitting_Test/031/pt_files/'      
SPLIT_VAL_PATH = '/home/anubis/c1307135/Grouped_sets/new_grouped/'

# In[]:
    
import torch.utils.data as data

IMG_EXTENSIONS = [".fits"]

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
    _data= no_nans(_data) 
    _data = numpy_transform(_data)
    _data = torch.from_numpy(_data)
    imager(_data)
    
    
    _label = file[0].header['KAPPA']
    file.close()

    if len(_data.shape) < 3:
        _data = _data.reshape((*_data.shape, 1))    
    return _data, _label


class FITSCubeDataset(data.Dataset):
    def __init__(self, data_path, cube_length, img_size, transforms):
        self.data_path = data_path
        self.transforms = transforms
        self.img_size = img_size
        self.cube_length = cube_length
        self.img_files = make_dataset(data_path)

    def __getitem__(self, index):
        cube_index = index // self.cube_length
        slice_index = index % self.cube_length  
        _img, _label = default_fits_loader(self.img_files[cube_index], self.img_size, slice_index)
        #_img[_img != _img] = 0  
        #_img = PIL_transform(_img)
        if self.transforms is not None:
            _data = (self.transforms(_img), _label)
               
        return _data

    def __len__(self):
        return len(self.img_files)*self.cube_length
        
# In[]:
def Validator(loader,device,loss):
    _val_accuracies = []        
    for idx, (batch, target) in enumerate(tqdm(loader)):
        batch = batch.to(device).to(torch.float)
        batch.view(batch_size,2,img_size[0],img_size[1])
        if isinstance(loss, torch.nn.CrossEntropyLoss):
            target = target.to(device).to(torch.long)
        else:
            target = target.to(device).to(torch.float)
        #pred = model(batch[:,0::2,:,:],batch[:,1::2,:,:]).reshape(-1)   
        pred = model(batch[:,0::2,:,:]).reshape(-1)
        loss_value = loss(pred, target)
        _val_accuracies.append(loss_value.detach())
          
    mean_val_accuracy = sum(_val_accuracies)/len(_val_accuracies)
    return(mean_val_accuracy)
    
def imager(image):
#    batch = batch.cpu().numpy()
#    for i in range(batch.shape[2]):
#    image = batch[i]
    plt.figure()
    plt.imshow(image[0],cmap='plasma')
    plt.colorbar()
    plt.savefig(OUTPUT_PATH + 'test.png')
    plt.close()
    plt.figure()
    plt.imshow(image[1],cmap='plasma')
    plt.colorbar()
    plt.savefig(OUTPUT_PATH + 'test2.png')
    plt.close()
        
def imager2(batch):
    index = np.arange(0,batch.shape[0],1)
    for i in range(batch.shape[0]):       
        image = batch[i,0,:,:]
        plt.figure()
        plt.imshow(image,cmap='plasma')
        plt.colorbar()
        plt.savefig(OUTPUT_PATH + str(index[i]) + 'test.png')
        plt.close()



        
# In[]:
        
"""
train_sampler, 

"""
       
def train(model: torch.nn.Module,
          transforms, 
          data_path= DATA_PATH, 
          val_path= VAL_PATH,
          split_val_path = SPLIT_VAL_PATH,
          num_epochs=1520, 
          batch_size=64, 
          verbose=True,
          cube_length=64,
          img_size=(64, 64), 
          loss=torch.nn.MSELoss(), 
          initial_lr=1e-4, 
          suffix=""):

    data_path = os.path.abspath(data_path)
    val_path = os.path.abspath(val_path)	
    model = model.train()
    device = torch.device("cuda")
    model = model.to(device).to(torch.float)
    
    """ LOADING IN THE REAL TRAINING AND VALIDATION DATASETS """
    
    loader = DataLoader(FITSCubeDataset(data_path, cube_length, transforms, img_size), 
                        batch_size=batch_size, shuffle = True)
                        #False, sampler=train_sampler
                          
    print('TRAINING DATA LOADED IN')
    validation_loader = DataLoader(FITSCubeDataset(val_path, cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
                                   #False, sampler=val_sampler
                                   
    loader_0 = DataLoader(FITSCubeDataset(split_val_path+'0/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
    loader_1 = DataLoader(FITSCubeDataset(split_val_path+'1/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
    loader_2 = DataLoader(FITSCubeDataset(split_val_path+'2/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)                                  
    loader_3 = DataLoader(FITSCubeDataset(split_val_path+'3/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
    loader_4 = DataLoader(FITSCubeDataset(split_val_path+'4/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
    loader_5 = DataLoader(FITSCubeDataset(split_val_path+'5/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
    loader_6 = DataLoader(FITSCubeDataset(split_val_path+'6/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)
    loader_7 = DataLoader(FITSCubeDataset(split_val_path+'7/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)                                   
    loader_8 = DataLoader(FITSCubeDataset(split_val_path+'8/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)                                   
    loader_9 = DataLoader(FITSCubeDataset(split_val_path+'9/', cube_length, transforms, img_size), 
                                   batch_size=batch_size, shuffle=True)    
                                   
    print('VALIDATION DATA LOADED IN')
                                                                        
    optim = torch.optim.Adam(model.parameters(), initial_lr)
    
    accuracies, epochs, val_epochs, bin_errs = [], [], [], []
    val_acc, running_val_acc = [], []
    
    for i in range(num_epochs):
        print("Epoch {0} of {1}" .format( (i+1), num_epochs))
        _accuracies,_val_accuracies = [],[]   
        model.train(True) 
        for idx, (batch, target) in enumerate(tqdm(loader)):
            batch = batch.to(device).to(torch.float)
            imager2(batch.cpu().numpy())
            #batch = batch.view(batch_size,2,img_size[0],img_size[1])
            dummy_batch = batch.cpu().numpy()
            
#            dummy_target = []
#            dummy_target.append(target.detach().numpy())
            if isinstance(loss, torch.nn.CrossEntropyLoss):
                target = target.to(device).to(torch.long)
            else:
                target = target.to(device).to(torch.float)
            #pred = model(batch[:,0::2,:,:],batch[:,1::2,:,:]).reshape(-1)
            #imager(batch)
            pred = model(batch[:,0::2,:,:]).reshape(-1)
            loss_value = loss(pred, target)
            optim.zero_grad()
	   
            ''' BACK PROPOGATE THE ERROR '''
            loss_value.backward()
            ''' LET THE OPTIMISER STEP FORWARD '''
            optim.step()
                
            ###Change the error metric here###

            _accuracies.append(loss_value.detach())
        
#            dummy_targets_ = np.hstack(dummy_target)
#            plt.figure()
#            plt.hist(dummy_target,bins=50)
#            plt.savefig(IMG_PATH+'histogram'+str(idx)+'.png')
        
        epochs.append(i+1)               
        mean_accuracy = sum(_accuracies)/len(_accuracies)
        accuracies.append(mean_accuracy)
        print("Mean training loss: %f" % mean_accuracy)        
        model.train(False)
        
        if i % 10 == 0:            
            val_epochs.append(i+1)
            grouped_val_acc = []
            running_average = []
            mean_val_accuracy = Validator(loader_0,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 0 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_1,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 1 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_2,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 2 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_3,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 3 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_4,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 4 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_5,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 5 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_6,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 6 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_7,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 7 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_8,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 8 COMPLETE: ',mean_val_accuracy)
            mean_val_accuracy = Validator(loader_9,device, loss)
            grouped_val_acc.append(mean_val_accuracy)
            running_average.append(mean_val_accuracy)
            print('GROUP 9 COMPLETE: ',mean_val_accuracy)
            
            val_acc.append(np.array(np.hstack(grouped_val_acc)))
            running_average.append(mean_val_accuracy)
            running_val_acc.append(np.mean(np.hstack(running_average)))
      
            '''
            for idx, (batch, target) in enumerate(tqdm(validation_loader)):
                batch = batch.to(device).to(torch.float)
                if isinstance(loss, torch.nn.CrossEntropyLoss):
                    target = target.to(device).to(torch.long)
                else:
                    target = target.to(device).to(torch.float)
                pred = model(batch).reshape(-1)   
                loss_value = loss(pred, target)
                  
                ###Change the error metric here###
    
                _val_accuracies.append(loss_value.detach())
                   
            mean_val_accuracy = sum(_val_accuracies)/len(_val_accuracies)
            val_accuracies.append(mean_val_accuracy)
            '''
            
            print("Mean Validation loss: %f" % np.mean(np.hstack(running_average))) 
        
        plot_accuracy_regression(accuracies,np.vstack(val_acc),running_val_acc,epochs, val_epochs, IMG_PATH, "Validation_loss%s.png" % suffix)        
        model.eval()        
        if i % 10 == 0:
            torch.save(model, MODEL_PATH+'Epoch_'+str(i)+'Regressional_Net%s.pt' % suffix )
            grouped_train_loss = np.vstack([epochs,accuracies])
            np.savetxt(IMG_PATH+'val_loss_values%s' % suffix,val_acc,delimiter=',')
            #grouped_val_loss = np.vstack([val_epochs,val_accuracies])
            np.savetxt(IMG_PATH+'loss_values%s' % suffix,grouped_train_loss,delimiter=',')
            #np.savetxt(IMG_PATH+'loss_values%s' % suffix,grouped_val_loss,delimiter=',')
        print('#'*73)
    torch.save(model, MODEL_PATH+'Regressional_Net_%s.pt' % suffix)
        
# In[]:
        
if __name__ == '__main__':
    
    print("Creating Model and Initializing weights")
	
#    for model_class, loss_fn, suffix in zip([CategoricalNet, RegressionNet], [torch.nn.CrossEntropyLoss(), torch.nn.MSELoss()], ["_categorical", "_regression"]):
#        for schedule in [True, False]:
            
    model_class, loss_fn, suffix = RegressionalNet, torch.nn.MSELoss(), "_031"
    schedule = False 
    model = model_class()
    model.apply(weight_init)
   # transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize((224,224)),transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    transform = transforms.Compose([transforms.Normalize([0.5], [0.5])])

start = time.time()
#print('#'*73)
#print('CREATING SAMPLE WEIGHTS')
#train_sampler_vals = weighting(DATA_PATH)
#train_sampler = WeightedRandomSampler(train_sampler_vals,num_samples = 16, replacement=True )
#print('SAMPLERS CREATED')
#print('#'*73)

#train_sampler,

train(model, transform, data_path= DATA_PATH, val_path= VAL_PATH,
      split_val_path = SPLIT_VAL_PATH, num_epochs=1, batch_size=16, cube_length=1,
      initial_lr=1e-3, loss=loss_fn, suffix=suffix)
      
#train(model, transform,  num_epochs=101, batch_size=64, lr_schedule=schedule, loss=loss_fn, suffix=suffix)
      
end = time.time()
print('TRAIN TIME:')
print('%.2gs'%(end-start))
print('#'*73)


