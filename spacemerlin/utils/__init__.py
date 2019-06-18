import os
from .sauron_colormap import sauron_cm, sauron_r_cm

_BASE_DATA_PATH = os.path.join(os.path.dirname(__file__), "data")

EXAMPLE_DATA_PATH = os.path.join(_BASE_DATA_PATH, "example_fits")
EXAMPLE_CHECKPOINT_PATH = os.path.join(_BASE_DATA_PATH, "checkpoints",
                                       "CAE_Epoch_300.pt")
EXAMPLE_PCA_PATH = os.path.join(_BASE_DATA_PATH, "checkpoints",
                                "PCA_routine.pkl")
EXAMPLE_RESULT_DIR = os.path.join(_BASE_DATA_PATH, "results", "CAE_Results.pkl")
DEFAULT_BOUNDARY = 2.960960960960961
