import io
import numpy as np
import matplotlib.pyplot as plt
from src.seed import SEEDLoader
from src.tuh import TUHLoader
from src.deap import DEAPLoader
from src.bci import BCILoader
from src.features import feature_extraction
from src.classifier import classifier
import pickle


dataset_name = "seed"
DESA = "desa1"
filterType = "filterbanks"
window_duration = 3 # in seconds
filterNo = 12

if dataset_name == "seed":
      dataset = SEEDLoader('/gpu-data3/ixour/seed', '/gpu-data3/ixour/seed/label.mat', idx=idx)
elif dataset_name == "tuh":
      dataset = TUHLoader('/gpu-data3/ixour/tuh/00_epilepsy/', '/gpu-data3/ixour/tuh/01_no_epilepsy/', preprocessed=True)
elif dataset_name == "deap":
      dataset = DEAPLoader('/gpu-data3/ixour/deap/data_preprocessed')
elif dataset_name == "bci":
      dataset = BCILoader("/gpu-data3/ixour/bci", preprocessed=True)


features = ['mean_iam', 'mean_ifm','var_ifm']

kwargs = {'window': window_duration * dataset.sfreq,
          'overlap': 0.5,
          'mode': 'linear'}

feature_matrix = feature_extraction(dataset, features, DESA, filterType=filterType, filterNo=filterNo, **kwargs)
print(feature_matrix.shape, feature_matrix[0]['feat'].shape)

if filterType == "filterbanks":
      np.save(f"/gpu-data3/ixour/{dataset_name}/featureMatrices/{DESA}_{filterNo}-{filterType}_{window_duration}s_{idx}.npy", feature_matrix)
else:
      np.save(f"/gpu-data3/ixour/{dataset_name}/featureMatrices/{DESA}-{filterType}_{window_duration}s_{idx}.npy", feature_matrix)


Check for NaN values
for sample in feature_matrix:
    for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']):
        bb = sample['feat'][i]  # 2D array
        # Find NaN indices in 2D
        nanIdx = np.argwhere(np.isnan(bb))
        if len(nanIdx) != 0:
            print("NaN indices:\n", nanIdx)

print("--- Feature Matrix shape ---")
print(f"Samples: {feature_matrix.shape[0]}\n"
      f"Bands x Channels x Features: {feature_matrix[0]['feat'].shape}")


##############################################################################################
###########################################  OTHER DATASETS  #################################

for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']):
      print(f"----- {band} Band -----")
      X = np.concatenate([[sample['feat'][i] for sample in feat] for feat in all_samples])
    #   print(f"Shape: {X.shape, y.shape}")
      results, accuracy = classifier(X, y, "svm")
      print(f"Accuracy score: {np.round(accuracy, 5) * 100} %")



##############################################################################################
###########################################  DEAP  ###########################################

all_samples = [feature_matrix]

y_valence, y_arousal = np.array([sample['label'][:2] for sample in feature_matrix]).T
y_valence = (y_valence > 5).astype(int)  
y_arousal = (y_arousal > 5).astype(int)  

for i, band in enumerate(['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma', 'All']):
      print(f"----- {band} Band -----")
      X = np.concatenate([[sample['feat'][i] for sample in feat] for feat in all_samples])
      results_valence, accuracy_valence = classifier(X, y_valence, "svm")
      results_arousal, accuracy_arousal = classifier(X, y_arousal, "svm")
      print("Accuracy score Valence-Arousal: " 
                  f"{np.round(accuracy_valence, 5) * 100}%-{np.round(accuracy_arousal, 5) * 100}%")


##############################################################################################

