import mne
import io
import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt


from src.loader import SEEDLoader

dataset = SEEDLoader('/gpu-data3/ixour/seed', '/gpu-data3/ixour/seed/label.mat')

from src.features import feature_extraction

# features = ['mean_iam', 'mean_ifm','var_ifm']
features = ['mean_iam']

# window = 200 (1sec)
kwargs = {'window': 200,
          'overlap': 0.5,
          'mode': 'linear'}

feature_matrix = feature_extraction(dataset, features, 'desa1', **kwargs)
np.save("featureMatrix_MIA.npy", feature_matrix)
