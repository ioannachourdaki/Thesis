import io
import numpy as np
import matplotlib.pyplot as plt
from src.seed import SEEDLoader
from src.tuh import TUHLoader
from src.deap import DEAPLoader
from src.bci import BCILoader
from src.features import feature_extraction
from src.classifier import classifier, crossval_classifier
import pickle


dataset_name = "deap"
DESA = "desa1"
filterType = "filterbanks"
window_duration = 10 # in seconds
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

feature_matrix, baselines = feature_extraction(dataset, features, DESA, filterType=filterType, filterNo=filterNo, **kwargs)
print(feature_matrix.shape, feature_matrix[0]['feat'].shape)

if filterType == "filterbanks":
      np.save(f"/gpu-data3/ixour/{dataset_name}/featureMatrices/{DESA}_{filterNo}-{filterType}_{window_duration}s_{idx}.npy", feature_matrix)
else:
      np.save(f"/gpu-data3/ixour/{dataset_name}/featureMatrices/{DESA}-{filterType}_{window_duration}s_{idx}.npy", feature_matrix)

targetType = "valence_arousal"
dataset_name = "deap"
DESA = "desa1"
filterNo = 12

for task in ["subject_independent"]:
      for filterType in ["filterbanks", "gabor"]:
            for window_duration in [10,3,1]:
                  if filterType == "filterbanks":
                        feature_matrix = np.load(f"/gpu-data3/ixour/{dataset_name}/featureMatrices/{DESA}_{filterNo}-{filterType}_{window_duration}s.npy", allow_pickle=True)
                  else:
                        feature_matrix = np.load(f"/gpu-data3/ixour/{dataset_name}/featureMatrices/{DESA}_{filterType}_{window_duration}s.npy", allow_pickle=True)
                  
                  print("-" * 100)
                  print(f"{filterType}-{window_duration}s")
                  print("-" * 100)
                  
                  multiclass = True if dataset_name == "seed" else False
                  
                  balanced_acc, roc_auc = crossval_classifier(feature_matrix, task, targetType, 'svm', multiclass)
                  balanced_acc = [[float(acc) for acc in row] for row in balanced_acc]
                  roc_auc = [[float(acc) for acc in row] for row in roc_auc]
                  print(f"{task}\n\tBalanced Accuracy: {balanced_acc}\n\tROC-AUC: {roc_auc}")
