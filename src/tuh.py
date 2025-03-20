import scipy.io
import mne
import numpy as np
import os
import pickle
import random
from src.utils import select_balanced_files, apply_ica
from tqdm import tqdm


class TUHLoader:
    def __init__(self, epilepsy_dir, no_epilepsy_dir, sfreq=250, selChannels=None, window_length=300, overlap=0.5, trim_length=1, preprocessed=False):
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
                         'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz'
                        ] # Exclude , 'Oz', 'A1', 'A2'
        self.selChannels = selChannels or self.channels
        if not preprocessed:
            self.selChannels.append('EKG')

        self.n_channels = len(self.selChannels)
        self.preprocessed = preprocessed
        self.data = self.load_dataset()


    def preprocessing(self, file_path):
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Exclude timestamps
        raw.set_meas_date(None)
        # Include recordings of duration 10-60 minutes
        raw_duration = (raw.get_data().shape[1] / self.sfreq) / 60
        if raw_duration < 10 or raw_duration > 60:
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

        raw.pick_channels([ch for ch in self.selChannels if ch in raw.ch_names], verbose=False)

        # Filter out f < 0.5Hz
        raw.filter(l_freq=0.5, h_freq=None, verbose=False)
        # Apply notch filter to remove 60 Hz (for power-line noise)
        raw.notch_filter(np.arange(60, 125, 60), verbose=False)
        # Remove artifacts and EKG channel
        if 'EKG' in raw.ch_names:
            raw = apply_ica(raw)

        new_file_path = file_path.replace(".edf", "_raw.fif")
        raw.save(new_file_path, overwrite=True)
        print("Preprocessed file is saved!")
        

    def get_segments(self, data):
        window = int(self.window_length * self.sfreq)
        step = int(window * (1 - self.overlap))
        result = np.lib.stride_tricks.sliding_window_view(data, window_shape=window, axis=1)[:, ::step]
        return np.transpose(result, axes=(1,0,2))
            

    def load_file(self, file_path, label):
        if not self.preprocessed:
            raw = self.preprocessing(file_path)
        else:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose=False)

        # Divide each sample in smaller segments
        eegs = []
        trim_size = int(self.trim_length * self.sfreq)
        segments = self.get_segments(raw.get_data()[:, trim_size:-trim_size])

        for segment in segments:
            eegs.append({'raw': mne.io.RawArray(segment, raw.info, verbose=False),
                         'label': label,
                         'subject': raw.info['subject_info']['his_id'],
                        })

        return eegs

    def load_dataset(self):
        all_eegs = []

        no_epilepsy_files = [os.path.join(self.no_epilepsy_dir, f) for f in os.listdir(self.no_epilepsy_dir)
                             if os.path.isfile(os.path.join(self.no_epilepsy_dir, f))] 
        
        if not self.preprocessed:
            # These files contain less channels, so they are not to be chosen
            exclude_files = [1375, 1357, 1199, 863, 796, 780, 743, 377]
            epilepsy_files,_ = select_balanced_files(self.epilepsy_dir, exclude_files, len(no_epilepsy_files))
        else: 
            epilepsy_files = [os.path.join(self.epilepsy_dir, f) for f in os.listdir(self.epilepsy_dir)
                              if os.path.isfile(os.path.join(self.epilepsy_dir, f))]

        for file in tqdm(no_epilepsy_files, desc="Loading No-Epilepsy Data"):
          all_eegs.extend(self.load_file(file, '0') or [])
        for file in tqdm(epilepsy_files, desc="Loading Epilepsy Data"):
          all_eegs.extend(self.load_file(file, '1') or [])

        random.shuffle(all_eegs)

        return all_eegs
