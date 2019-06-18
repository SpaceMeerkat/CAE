# -*- coding: utf-8 -*-
"""
Created on Mon May 13 09:31:52 2019

@author: SpaceMeerkat
"""

import os

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

from spacemerlin.data_loading import FITSCubeDataset
from spacemerlin.models import Autoencoder
from spacemerlin.utils import EXAMPLE_CHECKPOINT_PATH, EXAMPLE_PCA_PATH, \
    DEFAULT_BOUNDARY


def run(model_path=EXAMPLE_CHECKPOINT_PATH, data_path="examples",
        pca_path=EXAMPLE_PCA_PATH, boundary=DEFAULT_BOUNDARY, save_path=None):

    # load model and set to evaluation mode on cpu device
    model = Autoencoder()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    #  Load the data using a given batch size (default=64) and given number of
    # workers (default=16
    test_loader = DataLoader(dataset=FITSCubeDataset(data_path),
                             batch_size=64, num_workers=16, shuffle=True)

    # setup intermediate storing variables
    all_names = []
    all_features = []

    # Pass the loaded images through the CAE model and store the results
    for idx, (batch, names) in tqdm(enumerate(test_loader)):
        features, ind1, s1, ind2, s2 = model.encoder(batch.float())
        all_names.append(names)
        all_features.append(features.detach().numpy())

    all_features = np.hstack(all_features)
    all_names = np.hstack(all_names)

    with open(pca_path, 'rb') as f:
        pca_model = pickle.load(f)

    # Load the PCA routine and transform the 3D latent data for making
    # predictions of circularity
    pca_test_data = pca_model.transform(all_features)

    # Get a circularity prediction from the 3D latent positions
    latent_x = np.sqrt((pca_test_data[:, 0] ** 2) + (pca_test_data[:, 1] ** 2))
    latent_y = np.abs(pca_test_data[:, 2])
    classifications = np.ones(len(latent_x))
    zero_indices = np.where(latent_x < boundary)[0]
    classifications[zero_indices] = 0

    # store results in data frame
    df_data = np.vstack([all_names, all_features.T, classifications]).T

    results = pd.DataFrame(df_data)
    results.columns = ["Name", "L1", "L2", "L3", "Circularity"]
    results.set_index(keys="Name", drop=True, inplace=True)

    # store results in pickle file if save_path is available
    if save_path is not None:
        results.to_pickle(os.path.join(save_path, "CAE_results.pkl"))

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--modelpath", type=str,
                        default=EXAMPLE_CHECKPOINT_PATH,
                        help="A path to a model checkpoint file")
    parser.add_argument("-d", "--datapath", type=str, default="examples",
                        help="A path to a data directory")
    parser.add_argument("-p", "--pcapath", type=str, default=EXAMPLE_PCA_PATH,
                        help="A path to a pca checkpoint (as pickle file)")
    parser.add_argument("-b", "--boundary", type=float, default=DEFAULT_BOUNDARY,
                        help="The boundary value"),
    parser.add_argument("-s", "--savepath", type=str, default=None,
                        help="The path to save the results to")

    args = parser.parse_args()

    run(model_path=args.modelpath, data_path=args.datapath,
        pca_path=args.pcapath, boundary=args.boundary, save_path=args.savepath)
