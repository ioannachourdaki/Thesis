import scipy.io
import mne
import numpy as np
import os
from src.utils import apply_ica
from tqdm import tqdm


class BCILoader:
    def __init__(self, data_dir, sfreq=250, selChannels=None, window_length=20, overlap=0.5, trim_length=1, preprocessed=False):
        self.data_dir = data_dir
        self.sfreq = sfreq
        self.window_length = window_length
        self.overlap = overlap
        self.trim_length = trim_length
        self.channels = [
            'Fz', 'C3', 'Cz', 'C4', 'Pz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 
            'C1', 'C2', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'P2', 'POz'
            ]
        self.selChannels = selChannels or self.channels
        self.n_channels = len(self.selChannels)
        self.preprocessed = preprocessed
        self.data = self.load_dataset()

    
    def preprocessing(self, file_path):
        raw = mne.io.read_raw_gdf(file_path, preload=True, verbose=False)
        rename_dict = {
                        'EEG-Fz': 'Fz', 'EEG-C3': 'C3', 'EEG-Cz': 'Cz', 'EEG-C4': 'C4', 
                        'EEG-Pz': 'Pz','EEG-0': 'FC3', 'EEG-1': 'FC1', 'EEG-2': 'FCz', 
                        'EEG-3': 'FC2', 'EEG-4': 'FC4', 'EEG-5': 'C5', 'EEG-6': 'C1', 
                        'EEG-7': 'C2', 'EEG-8': 'C6', 'EEG-9': 'CP3', 'EEG-10': 'CP1', 
                        'EEG-11': 'CPz', 'EEG-12': 'CP2', 'EEG-13': 'CP4', 'EEG-14': 'P1', 
                        'EEG-15': 'P2', 'EEG-16': 'POz'
        }
        raw.rename_channels(rename_dict)
        raw.set_channel_types({'EOG-left': 'eog'})
        raw.set_channel_types({'EOG-central': 'eog'})
        raw.set_channel_types({'EOG-right': 'eog'})
        raw = apply_ica(raw)
        
        new_file_path = file_path.replace(".gdf", "") + "_raw.fif"
        raw.save(new_file_path, overwrite=True)
        print("Preprocessed file is saved!")
        return raw


    def get_segments(self, data):
        window = int(self.window_length * self.sfreq)
        step = int(window * (1 - self.overlap))
        result = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=1)[:, ::step]
        return np.transpose(result, axes=(1,0,2))


    def load_file(self, file_path):
        if not self.preprocessed:
            raw = self.preprocessing(file_path)
        else:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        trim_size = self.trim_length * self.sfreq

        # Select channels
        raw.pick_channels(self.selChannels, verbose=False)

        eegs = []

        segments = self.get_segments(raw.get_data()[:, trim_size:-trim_size])

        for segment in segments:
            eegs.append({'raw': mne.io.RawArray(segment, raw.info, verbose=False), 
                        'subject': int(file_path.split("/")[-1][2])
                        })

        return eegs

    def load_dataset(self):
        all_eegs = []
        files = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir)]

        for file in tqdm(files, desc="Loading Motor-Imagery Data"):
          all_eegs.extend(self.load_file(file))

        return all_eegs
