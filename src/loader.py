import os
import scipy.io
import mne


class SEEDLoader:
    def __init__(self, data_dir, label_file, sfreq=200, channels=None):
        """
        Initialize the SEEDDataset.

        Parameters:
        - data_dir: str, directory containing the .mat files.
        - label_file: str, path to the label.mat file.
        - sfreq: float, sampling frequency of the EEG data.
        - channels: list of str, names of EEG channels.
        """
        self.data_dir = data_dir
        self.labels = scipy.io.loadmat(label_file)['label'][0]
        self.sfreq = sfreq
        self.channels = channels or [
            'Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'Fz',
            'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2',
            'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4',
            'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'CP6', 'TP8', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6',
            'P8', 'PO7', 'PO5', 'PO3', 'POz', 'PO4', 'PO6', 'PO8', 'Cb1',
            'O1', 'Oz', 'O2', 'Cb2'
        ]
        self.n_channels = len(self.channels)
        self.data = self.load_dataset()

    def load_file(self, file_path):
        """
        Load all EEG signals and their labels from a single .mat file.

        Parameters:
        - file_path: str, path to the .mat file.

        Returns:
        - data: list of dicts, each containing 'raw' and 'label'.
        """
        info = mne.create_info(self.channels, sfreq=self.sfreq, ch_types=['eeg'] * self.n_channels)
        mat_data = scipy.io.loadmat(os.path.join(self.data_dir, file_path))

        # 15 EEGs of the subject on this day
        eegs = [{'raw': mne.io.RawArray(mat_data[f'djc_eeg{i+1}'] * 1e-6, info), 
                 'label': label,
                 'subject': int(file_path[0]),
                 'video': i+1
                 } 
                 for i, label in zip(range(15), self.labels)]
        return eegs

    def load_dataset(self):
        all_eegs = []
        mat_files = [f for f in os.listdir(self.data_dir) 
                     if f.endswith('.mat') and not f.startswith('label')]
        
        for mat_file in mat_files:
          all_eegs.extend(self.load_file(mat_file))
        return all_eegs
