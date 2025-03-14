import scipy.io
import mne
import numpy as np
import os
import pickle
from src.utils import z_score
from tqdm import tqdm


class DEAPLoader:
    def __init__(self, data_dir, sfreq=128, selChannels=None, window_length=15, overlap=0.5, trim_length=1):
        """
        Initialize the DEAPDataset.

        Parameters:
        - data_dir: str, directory containing the .mat files.
        - label_file: list, containing (valence, arousal, dominance, liking)
        - sfreq: float, sampling frequency of the EEG data.
        - channels: list of str, names of EEG channels.
        """
        self.data_dir = data_dir
        self.sfreq = sfreq
        self.window_length = window_length
        self.overlap = overlap
        self.trim_length = trim_length
        self.channels = [
            "Fp1", "AF3", "F3", "F7", "FC5", "FC1", "C3", "T7", "CP5", "CP1",
            "P3", "P7", "PO3", "O1", "Oz", "Pz", "Fp2", "AF4", "Fz", "F4",
            "F8", "FC6", "FC2", "Cz", "C4", "T8", "CP6", "CP2", "P4", "P8",
            "PO4", "O2"
        ]
        self.selChannels = selChannels or self.channels
        self.n_channels = len(self.selChannels)
        self.data = self.load_dataset()

    
    def get_segments(self, data):
        window = int(self.window_length * self.sfreq)
        step = int(window * (1 - self.overlap))
        result = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=1)[:, ::step]
        return np.transpose(result, axes=(1,0,2))


    def load_file(self, file_path):
        """
        Load all EEG signals and their labels from a single .mat file.

        Parameters:
        - file_path: str, path to the .mat file.

        Returns:
        - data: list of dicts, each containing 'raw' and 'label'.
        """
        info = mne.create_info(self.selChannels, sfreq=self.sfreq, ch_types=['eeg'] * self.n_channels, verbose=False)
        with open(os.path.join(self.data_dir, file_path), 'rb') as f:
            dat_data = pickle.load(f, encoding='latin1') 

        # norm_data = z_score(dat_data['data']) 
        norm_data = dat_data['data']

        trim_size = self.trim_length * self.sfreq

        # Select channels
        if self.n_channels != 32:
            idxChannels = [self.channels.index(channel) for channel in self.selChannels]

        eegs = []

        for trial, label in zip(norm_data, dat_data['labels']):
            if self.n_channels != 32:
                segments = self.get_segments(trial[idxChannels, trim_size:-trim_size])                 
            else:
                segments = self.get_segments(trial[:32, trim_size:-trim_size])
            
            for segment in segments:
                eegs.append({'raw': mne.io.RawArray(segment, info, verbose=False),
                            'label': label,
                            'subject': int(file_path[1:3])
                            }) #### ADD: 'physiological': trial[32:, trim_size:-trim_size] ###

        return eegs

    def load_dataset(self):
        all_eegs = []
        dat_files = [f for f in os.listdir(self.data_dir)]
        
        for dat_file in tqdm(dat_files, desc="Loading DEAP Data"):
          all_eegs.extend(self.load_file(dat_file))

        return all_eegs
