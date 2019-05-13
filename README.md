# **A Machine Learning Approach for Studying Galaxy Kinematics with ALMA in Preparation for the SKA**

This repository contains a set of Python scripts detailing a model for classifying galaxies as either kinematically disturbed, or disk-like and orderly rotating objects. The primary goal is that such a model can be used, or built upon, in order to maximise the number of irregular objects flagged by SKA in the future to ascertain cosmological truths (such as the merger rate of the universe at a given redshift) and aid current surveys involving kinematic modelling by returning fast predictions of galaxy properties.   
 
## **Using a convolutional autoencoder to return the level of ordered rotation of a galaxy**

Using the pre-trained [convolutional autoencoder](https://github.com/SpaceMeerkat/CAE/blob/master/Kinematics_Tester_Files/CAE_Epoch_300.pt), users can quickly output the latent  (CAE) positions of galaxies and recover the following information:

1. The circularity- defined as the amount of kinetic enrgy invested in flat circular rotation of a galaxy and thus the degree of kinematic disorder.

2. The galaxy position angle with an uncertainty of ~20 degrees (depending on the inclination).

## **Implementation information** 

The CAE has been created and trained using the [PyTorch](https://pytorch.org/) platform. In order to use the model, the user will be required to have PyTorch installed but the use of GPU capabilities is not a requirement.

Other dependencies:

* [Python v3.5 or later](https://www.python.org/)
* [PyTorch v1.1](https://pytorch.org/)
* [Astropy v3.1.2](https://www.astropy.org/)
* [NumPy v1.16.2](https://www.numpy.org/)

Additional optional dependencies:

* [Pandas v0.24.2](https://pandas.pydata.org/)


