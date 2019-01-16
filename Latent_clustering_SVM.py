#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 11:01:01 2019

@author: jamesdawson
"""


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd 
plt.close('all')
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn import svm
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier
from scipy.stats import binned_statistic_2d
plt.close('all')
from sklearn.metrics import confusion_matrix

#=============================================================================#

### LOADING IN THE DATA

path = '/home/jamesdawson/Documents/'
df = pd.read_pickle(path+'features_train.pkl')
features = df.values[:,:-1]
labels_train = df.values[:,-1]
copy_labels = np.copy(labels_train)

df2 = pd.read_pickle(path+'features_test.pkl')
features2 = df2.values[:,:-1]
labels_test = df2.values[:,-1]

f_train = features
f_test = features2

thresh = 5
#thresh2 = 7.5

weights_train = 1./labels_train
labels_train = labels_train*10
labels_train[labels_train<thresh] = 0
#labels_train[(thresh<labels_train) & (labels_train<=thresh2)] = 1
labels_train[labels_train>thresh] = 1

ax2_labels = np.copy(labels_test)
labels_test = labels_test*10
labels_test[labels_test<thresh] = 0
#labels_test[(thresh<labels_test) & (labels_test<=thresh2)] = 1
labels_test[labels_test>thresh] = 1
#=============================================================================#

### PERFORMING THE 10 Dimensional SVM FITTING ROUTINE

clf = svm.SVC(kernel='poly',C = 1e-3 ,class_weight='balanced')
clf.fit(f_train, labels_train)

trains = clf.predict(f_train)
confidence = accuracy_score(trains,labels_train)
print('Training confidence: %.2f' % confidence)

tests = clf.predict(f_test)
confidence = accuracy_score(labels_test,tests)
print('Testing confidence: %.2f' %confidence)

hard_pred = clf.predict(features2)

#=============================================================================#

### CONFUSION MATRIX

cfm = confusion_matrix(labels_test,tests)
cm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]

print(cm)

#tn, fp, fn, tp = cm.ravel()
#
#print("tn, fp, fn, tp",(tn, fp, fn, tp))

#=============================================================================#

### PLOTTING 2D PCA DATA AND DECISION BOUNDARY

pca = PCA(n_components=2)
PCA = pca.fit(features)

f_train_2d =  PCA.transform(features)
f_test_2d = PCA.transform(features2)

clf = svm.SVC(kernel='poly',C = 1e-3 ,class_weight='balanced')
clf.fit(f_train_2d, labels_train)


plt.figure(figsize=(10,10))
plt.subplot(211)

plt.text(20,15,s='Train',bbox=dict(facecolor='white', alpha=0.5))

plt.scatter(f_train_2d[:,0],f_train_2d[:,1],c=copy_labels,s=0.5)
plt.colorbar(label='$\kappa$')
plt.xlabel('PC-1')
plt.ylabel('PC-2')

ax = plt.gca()
xlim1 = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim1[0], xlim1[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
           linestyles=['-'])

plt.subplot(212)

plt.text(15,15,s='Test',bbox=dict(facecolor='white', alpha=0.5))
plt.scatter(f_test_2d[:,0],f_test_2d[:,1],c=ax2_labels,s=4)
plt.colorbar(label='$\kappa$')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()
xx = np.linspace(xlim[0], xlim[1], 30)
yy = np.linspace(ylim[0], ylim[1], 30)
YY, XX = np.meshgrid(yy, xx)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
Z = clf.decision_function(xy).reshape(XX.shape)
ax.contour(XX, YY, Z, colors='k', levels=[0], alpha=0.5,
           linestyles=['-'])

plt.xlabel('PC-1')
plt.ylabel('PC-2')


#=============================================================================#

### ROTATING THE HYPERPLANE BY THE BOUNDARY LINE ANGLE

#w = clf.coef_[0]
#a = -w[0] / w[1]
#xx = np.array(xlim1)
#yy = a * xx - (clf.intercept_[0]) / w[1]
#
#boundary = np.vstack([xx,yy])

angle = -0.2355

import math

def rotate_origin_only(xy, radians):
    """Only rotate a point around the origin (0, 0)."""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy


xy = np.vstack([f_train_2d[:,0],f_train_2d[:,1]])

rot_x, rot_y = rotate_origin_only(xy ,angle*np.pi) # Coordinates of the rotated training data
#b_x, b_y = rotate_origin_only(boundary, angle*np.pi) # Coordinates of the rotated boundary line

#=============================================================================#

### EVALUATING THE ROTATED DATA VIA A CONCENTRATION HISTOGRAM

fraction = lambda a, threshold: len(a[a<threshold])/len(a)

H, xedges, yedges, binnumber = binned_statistic_2d(np.array(rot_x).reshape(-1),np.array(rot_y).reshape(-1),
                                                   copy_labels,statistic=lambda a: fraction(a, thresh/10),bins=50)

XX, YY = np.meshgrid(xedges, yedges)
H_plotter = np.ma.masked_invalid(H)
H_flipped = H.T

plt.figure(figsize=(10,7))
ax1=plt.subplot(111)
plot1 = ax1.pcolormesh(XX,YY,H_plotter.T,vmin=np.nanmin(H_flipped), vmax=np.nanmax(H_flipped))
cbar = plt.colorbar(plot1,ax=ax1,label='$ N_{\kappa < 0.5}/N_{total}$')
#plt.plot(np.array(b_x).reshape(-1),np.array(b_y).reshape(-1),'r--',label='Boundary line')
plt.xlabel('Rotated hyperplane 1')
plt.ylabel('Rotated hyperplane 2')
plt.xlim(xedges.min(),xedges.max())
plt.ylim(yedges.min(),yedges.max())
plt.legend(loc='lower right')


xy_test = np.vstack([f_test_2d[:,0],f_test_2d[:,1]])
rot_x, rot_y = rotate_origin_only(xy_test ,angle*np.pi)
xy_test = np.vstack([rot_x,rot_y])
xy,xy_transformed = f_test,xy_test

plt.scatter(xy_transformed[0],xy_transformed[1],c=ax2_labels,marker='+',cmap='autumn',alpha=0.5)
plt.colorbar(label='$\kappa$')


test_x = xy_transformed[0]
test_y = xy_transformed[1]
x_int = np.searchsorted(xedges, test_x, side='left')-1
x_int[x_int==-1]=0
x_int[x_int==50]=49
y_int = np.searchsorted(yedges, test_y, side='left')-1      
y_int[y_int==-1]=0
y_int[y_int==50]=49

p_values = H_flipped[[y_int],[x_int]].data

grouped_prediction = pd.DataFrame(np.vstack([ax2_labels,labels_test,hard_pred,p_values]).T)
grouped_prediction.columns = ['Label', 'Binary_Label', 'Predicted_Label', 'Concentration']

#=============================================================================#

### TESTING THE ACCURACY OF THE CONCENTRATION SCORE

below = grouped_prediction.loc[grouped_prediction['Predicted_Label'] == np.nan] # Rows predicted as k>0.5
below2 = below.loc[below['Label'] > 0.5] # Rows predicted as 0 with kappa>0.5
threshold = below2.loc[below2['Concentration'] < 0.5] # Rows of the above with P(k>0.5) < 0.5
grouped_prediction.loc[threshold.index.values,'Predicted_Label'] = 1

#below = grouped_prediction.loc[grouped_prediction['Predicted_Label'] == 0] # Rows predicted as k>0.5
#below2 = below.loc[below['Label'] > 0.5] # Rows predicted as 0 with kappa>0.5
#threshold = below2.loc[below2['Concentration'] < 0.5] # Rows of the above with P(k>0.5) < 0.5
#grouped_prediction.loc[threshold.index.values,'Predicted_Label'] = 1
#
#below = grouped_prediction.loc[grouped_prediction['Predicted_Label'] == 0] # Rows predicted as k>0.5
#below2 = below.loc[below['Label'] > 0.5] # Rows predicted as 0 with kappa>0.5
#threshold = below2.loc[below2['Concentration'] < 0.5] # Rows of the above with P(k>0.5) < 0.5
#grouped_prediction.loc[threshold.index.values,'Predicted_Label'] = 1
#
#below = grouped_prediction.loc[grouped_prediction['Predicted_Label'] == 1] # Rows predicted as k>0.5
#below2 = below.loc[below['Label'] < 0.5] # Rows predicted as 0 with kappa>0.5
#threshold = below2.loc[below2['Concentration'] > 0.5] # Rows of the above with P(k>0.5) < 0.5
#grouped_prediction.loc[threshold.index.values,'Predicted_Label'] = 0
#
#below = grouped_prediction.loc[grouped_prediction['Predicted_Label'] == 1] # Rows predicted as k>0.5
#below2 = below.loc[below['Label'] < 0.5] # Rows predicted as 0 with kappa>0.5
#threshold = below2.loc[below2['Concentration'] > 0.5] # Rows of the above with P(k>0.5) < 0.5
#grouped_prediction.loc[threshold.index.values,'Predicted_Label'] = 0
#

#cfm = confusion_matrix(grouped_prediction['Binary_Label'],grouped_prediction['Predicted_Label'])
#cm = cfm.astype('float') / cfm.sum(axis=1)[:, np.newaxis]
#
#print(cm)

#=============================================================================#

from sklearn.metrics import roc_curve, auc

fpr, tpr, _ = roc_curve(grouped_prediction['Binary_Label'], grouped_prediction['Predicted_Label'])
roc_auc = auc(fpr, tpr)

print('ROC_AUC score: ',roc_auc)

plt.figure()
lw = 2
plt.plot(fpr, tpr, color='navy',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='grey', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()
