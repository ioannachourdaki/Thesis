import scipy.io
import mne
import numpy as np
import os
import pickle
import random
from tqdm import tqdm


class TUHLoader:
    def __init__(self, epilepsy_dir, no_epilepsy_dir, sfreq=250, selChannels=None, window_length=20, overlap=0.5, trim_length=1):
        """
        Initialize the SEEDDataset.

        Parameters:
        - data_dir: str, directory containing the .mat files.
        - label_file: str, path to the label.mat file.
        - channels: list of str, names of EEG channels.
        """
        self.epilepsy_dir = epilepsy_dir
        self.no_epilepsy_dir = no_epilepsy_dir
        self.sfreq = sfreq
        self.window_length = window_length
        self.overlap = overlap
        self.trim_length = trim_length
        self.channels = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'EKG'
                        ] # Exclude , 'Oz', 'A1', 'A2'
        self.selChannels = selChannels or self.channels
        if 'EKG' not in self.selChannels:
            self.selChannels.append('EKG')

        self.n_channels = len(self.selChannels)
        self.data = self.load_dataset()


    def get_segments(self, data):
        window = int(self.window_length * self.sfreq)
        step = int(window * (1 - self.overlap))
        result = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=1)[:, ::step]
        return np.transpose(result, axes=(1,0,2))
    

    def apply_ica(self, raw):
        ica = mne.preprocessing.ICA(method='fastica', max_iter=1000, random_state=42, verbose=False)
        ica.fit(raw, verbose=False)  
        # Find ECG artifacts
        ecg_inds, scores = ica.find_bads_ecg(raw, ch_name='EKG', method='correlation', verbose=False)
        # Remove identified ECG components
        ica.exclude = ecg_inds
        # Apply ICA to clean EEG data
        ica.apply(raw, verbose=False)
        # Remove EKG channel
        return raw.pick_types(eeg=True, verbose=False)
            

    def load_file(self, file_path, label):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Exclude recordings of duration under 1 minute
        if raw.n_times / raw.info['sfreq'] < 60:
            return None

        # Rename channels
        rename_dict = {ch: ch.replace("EEG ", "")
                             .replace("-LE", "")
                             .replace("-REF", "")
                             .replace("Z", "z")
                             .replace("FP", "Fp")
                             .replace("EKG1", "EKG")
                             .replace("PULSE", "EKG") 
                             .replace("ECG EKG-REF", "EKG")
                       for ch in raw.ch_names}
            
        raw.rename_channels(rename_dict)

        if 'EKG' in raw.ch_names:
            raw.set_channel_types({'EKG': 'ecg'})
        
        ################## DELETE #######################
        for ch in self.selChannels:
            if ch not in raw.ch_names and ch!="EKG":
                print(f"Channel not in raw: {ch}")
        ################## DELETE #######################

        raw.pick_channels([ch for ch in self.selChannels if ch in raw.ch_names], verbose=False)

        # Filter out f < 0.5Hz
        raw.filter(l_freq=0.5, h_freq=None, verbose=False)
        # Apply notch filter to remove 60 Hz (for power-line noise)
        raw.notch_filter(np.arange(60, 125, 60), verbose=False)
        # Remove artifacts and EKG channel
        if 'EKG' in raw.ch_names:
            raw = self.apply_ica(raw)

        # Divide each sample in smaller segments
        eegs = []
        trim_size = int(self.trim_length * raw.info['sfreq'])
        segments = self.get_segments(raw.get_data()[:, trim_size:-trim_size])

        ################## DELETE THIS!!! ##################
        unique_shapes = set(seg.shape for seg in segments)
        for ushape in unique_shapes:
            if ushape != (19, 5000):
                print(f"Different shape: {unique_shapes}")
        ######################################################

        for segment in segments:
            eegs.append({'raw': mne.io.RawArray(segment, raw.info, verbose=False),
                         'label': label,
                         'subject': raw.info['subject_info']['his_id'],
                        })

        return eegs

    def load_dataset(self):
        all_eegs = []

        no_epilepsy_files = [os.path.join(self.no_epilepsy_dir, f) for f in os.listdir(self.no_epilepsy_dir)] 
        epilepsy_files = [os.path.join(self.epilepsy_dir, f) for f in os.listdir(self.epilepsy_dir)] 
        epilepsy_files = epilepsy_files[:376] + epilepsy_files[377:600] ######## CHANGE THIS #######

        for edf_file in tqdm(no_epilepsy_files, desc="Loading No-Epilepsy Data"):
          all_eegs.extend(self.load_file(edf_file, '0') or [])
        for edf_file in tqdm(epilepsy_files, desc="Loading Epilepsy Data"):
          all_eegs.extend(self.load_file(edf_file, '1') or [])
        
        random.shuffle(all_eegs)

        with open('/gpu-data3/ixour/tuh/data_preprocessed/all_eegs.pkl', 'wb') as f:
            pickle.dump(all_eegs, f)
        
        print("Dataset is saved!")

        return all_eegs
