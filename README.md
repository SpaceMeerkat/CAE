# **ALMA/SKA Machine Learning Pipeline Tool**

This repository contains a set of Python scripts detailing a model for classifying galaxies as either kinematically disturbed, or disk-like and orderly rotating objects. The primary goal is that such a model can be used, or built upon, in order to maximise the number of irregular objects flagged by SKA in the future to ascertain cosmological truths (such as the merger rate of the universe at a given redshift) and aid current surveys involving kinematic modelling by returning fast predictions of galaxy properties.   
 
## **Using a convolutional autoencoder to return the level of ordered rotation of a galaxy**

Using the pre-trained [convolutional autoencoder](https://github.com/SpaceMeerkat/CAE/blob/master/Kinematics_Tester_Files/CAE_Epoch_300.pt), users can quickly output the latent  (CAE) positions of galaxies and recover the following information:

1. The circularity- defined as the amount of kinetic enrgy invested in flat circular rotation of a galaxy and thus the degree of kinematic disorder.

2. The galaxy position angle with an uncertainty of ~20 degrees (depending on the inclination).

## **Implementation information** 

The CAE has been created and trained using the [PyTorch](https://pytorch.org/) platform. In order to use the model, the user will be required to have PyTorch installed but the use of GPU capabilities is not a requirement.

Other **dependencies**:

* [Python v3.5 or later](https://www.python.org/)
* [PyTorch v1.1](https://pytorch.org/)
* [Astropy v3.1.2](https://www.astropy.org/)
* [NumPy v1.16.2](https://www.numpy.org/)

Additional optional dependencies:

* [Pandas v0.24.2](https://pandas.pydata.org/)

To use the model, one should alter the **example_script.py** file so that the paths to the model pt file and relevant modules in **testing_modules.py** are correctly imported. The example script also outputs a pandas DataFrame to a .pkl file should the user wish to save their results. 

## **Breakdown of the repository structure**

-The [Kinematics_Tester_Files](https://github.com/SpaceMeerkat/CAE/tree/master/Kinematics_Tester_Files) folder contains example scripts for using the CAE model.

-The [Test_FITS_files](https://github.com/SpaceMeerkat/CAE/tree/master/Test_FITS_files) folder has a selection of 30 ALMA velocity maps that can be used with the model for testing purposes. For more information on the galaxies used please read the corresponding journal papers [Zabel et al. 2019](https://academic.oup.com/mnras/article/483/2/2251/5218520) and [WISDOM project](https://academic.oup.com/mnras/article/468/4/4663/3072185).
