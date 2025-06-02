"""
CleanEEG - Automated Python-based Resting-State EEG Preprocessing GUI

Author: Amin Kabir
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
import numpy as np
from PyQt5.QtWidgets import (QApplication, QWidget, QFileDialog,
                             QMessageBox, QListWidgetItem, QLabel, QPushButton, QVBoxLayout)
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QObject, QTimer
from PyQt5.QtGui import QDoubleValidator, QIntValidator
from PyQt5 import uic
import mne
from mne.preprocessing import ICA
from mne_icalabel import label_components
from meegkit import dss
from meegkit.asr import ASR
from pyprep.find_noisy_channels import NoisyChannels
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5 import NavigationToolbar2QT
import warnings
from scipy.io import loadmat
from scipy import signal
matplotlib.use('Qt5Agg')
warnings.filterwarnings('ignore')


class MNEPlotWidget(QWidget):
    """Custom widget to embed MNE plots in the GUI"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(12, 6))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def clear(self):
        """Clear the plot"""
        self.figure.clear()
        self.canvas.draw()


class CleanEEGWorker(QThread):
    """Worker thread for EEG preprocessing to keep UI responsive"""
    progress_update = pyqtSignal(int)
    status_update = pyqtSignal(str)
    processing_complete = pyqtSignal(str, object)  # filename, processed_raw
    processing_error = pyqtSignal(str, str)  # filename, error_message

    def __init__(self, file_path: str, output_path: str, settings: Dict[str, Any]):
        super().__init__()
        self.file_path = file_path
        self.output_path = output_path
        self.settings = settings
        self.ica_object = None
        self.processing_stages = {}  # Store raw data at each stage for quality assessment
        self.snr_log = {}  # Store SNR values at each stage

    @staticmethod
    def _read_mat_locations(fname):
        """Read channel locations from a Brainstorm .mat file"""
        mat = loadmat(fname)
        if 'Channel' not in mat:
            raise ValueError('MAT file does not contain "Channel" key.')
        channel_data = mat['Channel'][0]
        ch_pos = {}
        for ch in channel_data:
            name = ch['Name'][0]
            loc = ch['Loc'].flatten() if ch['Loc'].shape == (3, 1) else ch['Loc']
            if abs(loc[0]) > 0.5 or abs(loc[1]) > 0.5 or abs(loc[2]) > 0.5:
                loc[0] = loc[0] / 1000.0
                loc[1] = loc[1] / 1000.0
                loc[2] = loc[2] / 1000.0
            ch_pos[name] = [loc[1], loc[0], loc[2]]
        return ch_pos

    @staticmethod
    def _read_ced_locations(fname):
        """Read channel locations from an EEGLAB .ced file"""
        ch_pos = {}
        if not os.path.isfile(fname):
            raise FileNotFoundError(f"The file {fname} does not exist.")
        with open(fname, 'r') as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("The .ced file is empty.")
        header_line = lines[0].strip()
        if not header_line:
            raise ValueError("The .ced file does not contain a header line.")
        parts = header_line.split('\t')
        if len(parts) < 4:
            parts = header_line.split()
        col_map = {col.strip().lower(): idx for idx, col in enumerate(parts)}
        required_columns = ['labels', 'x', 'y', 'z']
        missing_cols = [col for col in required_columns if col not in col_map]
        if missing_cols:
            raise ValueError(f"Missing required columns in header: {missing_cols}")
        label_idx = col_map['labels']
        x_idx = col_map['x']
        y_idx = col_map['y']
        z_idx = col_map['z']
        for line_num, line in enumerate(lines[1:], start=2):
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) <= max(label_idx, x_idx, y_idx, z_idx):
                parts = line.split()
            if len(parts) <= max(label_idx, x_idx, y_idx, z_idx):
                raise ValueError(f"Invalid line in CED file at line {line_num} (not enough columns): {line}")
            label = parts[label_idx].strip()
            try:
                x = float(parts[x_idx])
                y = float(parts[y_idx])
                z = float(parts[z_idx])
            except ValueError:
                raise ValueError(f"Invalid numerical values in line {line_num}: {line}")
            if abs(x) > 0.5 or abs(y) > 0.5 or abs(z) > 0.5:
                x = x / 1000.0
                y = y / 1000.0
                z = z / 1000.0
            ch_pos[label] = np.array([y, x, z])
        return ch_pos

    @staticmethod
    def compute_snr(raw: mne.io.Raw, freq_bands: Optional[Dict[str, Tuple[float, float]]] = None, method: str = 'rms')\
            -> Dict[str, Dict[str, float]]:
        """
        Compute Signal-to-Noise Ratio for EEG data.

        Parameters:
        -----------
        raw : mne.io.Raw
            The EEG data
        freq_bands : dict
            Dictionary of frequency bands to analyze
        method : str
            Method for SNR calculation ('rms' or 'spectral')

        Returns:
        --------
        snr_results : dict
            SNR values for different frequency bands
        """
        if freq_bands is None:
            freq_bands = {
                'delta': (1, 4),
                'theta': (4, 8),
                'alpha': (8, 13),
                'beta': (13, 30),
                'gamma': (30, 100)
            }

        # Get data and sampling frequency
        data = raw.get_data()
        sfreq = raw.info['sfreq']

        snr_results = {}

        if method == 'spectral':
            # Compute PSD
            freqs, psd = signal.welch(data, sfreq, nperseg=int(2*sfreq))

            for band_name, (low_freq, high_freq) in freq_bands.items():
                # Find frequency indices
                freq_mask = (freqs >= low_freq) & (freqs <= high_freq)

                # Signal power in the band
                signal_power = np.mean(psd[:, freq_mask], axis=1)

                # Noise estimation (neighboring frequencies)
                noise_low = max(0, low_freq - 2)
                noise_high = min(freqs[-1], high_freq + 2)
                noise_mask = ((freqs >= noise_low) & (freqs < low_freq)) | ((freqs > high_freq) & (freqs <= noise_high))

                if np.any(noise_mask):
                    noise_power = np.mean(psd[:, noise_mask], axis=1)
                    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
                else:
                    snr = np.full(len(raw.ch_names), np.nan)

                snr_results[band_name] = {
                    'mean_snr': np.nanmean(snr),
                    'std_snr': np.nanstd(snr),
                    'channel_snr': snr
                }

        else:  # RMS method
            for band_name, (low_freq, high_freq) in freq_bands.items():
                # Skip bands that exceed Nyquist frequency
                if high_freq > sfreq / 2:
                    snr_results[band_name] = {
                        'mean_snr': np.nan,
                        'std_snr': np.nan,
                        'channel_snr': np.full(len(raw.ch_names), np.nan)
                    }
                    continue

                # Filter data to frequency band
                raw_filtered = raw.copy().filter(low_freq, high_freq, verbose=False)
                filtered_data = raw_filtered.get_data()

                # RMS of signal
                signal_rms = np.sqrt(np.mean(filtered_data**2, axis=1))

                # Estimate noise from high frequencies (above 80 Hz)
                if raw.info['sfreq'] > 160:  # Ensure we can filter above 80 Hz
                    raw_noise = raw.copy().filter(80, None, verbose=False)
                    noise_data = raw_noise.get_data()
                    noise_rms = np.sqrt(np.mean(noise_data**2, axis=1))
                    snr = 20 * np.log10(signal_rms / (noise_rms + 1e-10))
                else:
                    # Use standard deviation as noise estimate
                    noise_std = np.std(filtered_data, axis=1)
                    snr = 20 * np.log10(signal_rms / (noise_std + 1e-10))

                snr_results[band_name] = {
                    'mean_snr': np.mean(snr),
                    'std_snr': np.std(snr),
                    'channel_snr': snr
                }

        return snr_results

    def run(self):
        try:
            filename = Path(self.file_path).name
            self.status_update.emit(f"Processing {filename}...")

            # Load the data
            self.progress_update.emit(5)
            raw_eeg = self._load_eeg_data(self.file_path)
            raw_original_for_report = raw_eeg.copy()
            bads_before_processing = []

            # Store original and compute initial SNR
            self.processing_stages['Original'] = raw_eeg.copy()
            self.snr_log['Original'] = self.compute_snr(raw_eeg)

            # Initialize progress tracking
            total_steps = sum([
                self.settings.get('set_montage', False),
                self.settings.get('apply_downsample', False),
                self.settings.get('remove_line_noise', False),
                self.settings.get('apply_bandpass', False),
                self.settings.get('detect_bad_channels', False),
                self.settings.get('apply_ica', False),
                self.settings.get('apply_asr', False),
                self.settings.get('interpolate_bads', False)
            ])

            if total_steps == 0:
                self.progress_update.emit(100)
                self.processing_complete.emit(filename, raw_eeg)
                return

            current_step = 0

            # Set montage (only if checkbox is checked and montage is provided)
            if self.settings.get('set_montage') and self.settings.get('montage'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Setting montage...")
                self._set_montage(raw_eeg, self.settings['montage'])
                # Montage doesn't change the signal, so no new stage needed

            # Resample (only if checkbox is checked)
            if self.settings.get('apply_downsample') and self.settings.get('resample_freq'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Downsampling data...")
                raw_eeg.resample(self.settings['resample_freq'])
                self.processing_stages['After Downsample'] = raw_eeg.copy()
                self.snr_log['After Downsample'] = self.compute_snr(raw_eeg)

            raw_clean = raw_eeg.copy()

            # Line noise removal (only if checkbox is checked)
            if self.settings.get('remove_line_noise'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Removing line noise...")
                raw_clean = self._remove_line_noise(raw_clean, self.settings['line_freq'])
                self.processing_stages['After Line Noise'] = raw_clean.copy()
                self.snr_log['After Line Noise'] = self.compute_snr(raw_clean)

            # Bandpass filter (only if checkbox is checked)
            if self.settings.get('apply_bandpass') and self.settings.get('highpass_freq'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Applying bandpass filter...")
                hp_freq = self.settings['highpass_freq']
                lp_freq = self.settings.get('lowpass_freq', 100)  # Default to 100 Hz if not specified
                raw_clean.filter(l_freq=hp_freq, h_freq=lp_freq, method='fir', verbose=False)
                self.processing_stages['After Bandpass'] = raw_clean.copy()
                self.snr_log['After Bandpass'] = self.compute_snr(raw_clean)

            # Bad channel detection (only if checkbox is checked)
            if self.settings.get('detect_bad_channels'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Detecting bad channels...")
                bad_channels = self._detect_bad_channels(raw_clean)
                raw_clean.info['bads'] = bad_channels
                bads_before_processing = list(bad_channels)
                self.processing_stages['After Bad Channels'] = raw_clean.copy()
                self.snr_log['After Bad Channels'] = self.compute_snr(raw_clean)

            # Pick channels
            raw_clean.pick(picks='eeg')

            # Set reference
            raw_clean.set_eeg_reference('average')
            raw_clean.apply_proj()

            # ICA (only if checkbox is checked)
            if self.settings.get('apply_ica'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Running ICA...")
                raw_clean, self.ica_object = self._apply_ica(raw_clean, self.settings)
                self.processing_stages['After ICA'] = raw_clean.copy()
                self.snr_log['After ICA'] = self.compute_snr(raw_clean)

            # ASR (only if checkbox is checked)
            if self.settings.get('apply_asr'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Applying ASR...")
                asr_cutoff = self.settings.get('asr_cutoff', 5)
                asr_calibration = self.settings.get('asr_calibration', 60)
                raw_clean = self._apply_asr(raw_clean, asr_cutoff, asr_calibration)
                self.processing_stages['After ASR'] = raw_clean.copy()
                self.snr_log['After ASR'] = self.compute_snr(raw_clean)

            # Interpolate bad channels (only if checkbox is checked and bad channels exist)
            if self.settings.get('interpolate_bads') and len(raw_clean.info['bads']) > 0:
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Interpolating bad channels...")
                raw_clean.interpolate_bads(reset_bads=True)
                self.processing_stages['After Interpolation'] = raw_clean.copy()
                self.snr_log['After Interpolation'] = self.compute_snr(raw_clean)

            # Store final processed data
            self.processing_stages['Final'] = raw_clean.copy()
            self.snr_log['Final'] = self.compute_snr(raw_clean)

            # Save the processed data
            self.progress_update.emit(95)
            self.status_update.emit("Saving processed data...")
            output_file = self._save_processed_data(raw_clean, filename)

            # Generate and save report if requested
            if self.settings.get('export_report'):
                self.status_update.emit("Generating enhanced HTML report...")
                self._generate_enhanced_report(
                    raw_original_for_report, raw_clean, bads_before_processing, filename,
                    self.settings.get('input_base_dir'))
                self.status_update.emit("Enhanced HTML report generated.")

            self.progress_update.emit(100)
            self.processing_complete.emit(filename, raw_clean)

        except Exception as e:
            self.processing_error.emit(filename, str(e))

    @staticmethod
    def _load_eeg_data(file_path: str) -> mne.io.Raw:
        """Load EEG data with automatic format detection and correct common mislabels."""
        try:
            raw = mne.io.read_raw(file_path, preload=True, verbose="WARNING")
            # Patterns for channels that aren’t true EEG
            eog_patterns = ["EOG", "VEOG", "HEOG", "EOGH", "EOGV", "EOG1", "EOG2", "LHEOG", "RHEOG"]
            ref_patterns = ["REF", "GND", "A1", "A2", "M1", "M2", "TP9", "TP10"]
            other_patterns = ["ECG", "EMG", "RESP", "TRIG", "STI"]
            mislabeled = []
            for ch in raw.ch_names:
                cu = ch.upper()
                if any(pat in cu for pat in eog_patterns):
                    mislabeled.append((ch, "eog"))
                elif any(pat in cu for pat in ref_patterns):
                    mislabeled.append((ch, "misc"))
                elif any(pat in cu for pat in other_patterns):
                    mislabeled.append((ch, "misc"))
            if mislabeled:
                for name, ctype in mislabeled:
                    print(f"Channel types corrected: {name} → {ctype}")
                    raw.set_channel_types({name: ctype})
            return raw

        except Exception as e:
            ext = Path(file_path).suffix.lower()
            raise ValueError(
                f"Failed to load '{file_path}' (.{ext}): {e}"
            )

    def _set_montage(self, raw: mne.io.Raw, montage_info: str):
        """Set EEG montage"""
        montage_type = self.settings.get('montage_type', 'template')
        montage_identifier = montage_info
        montage_applied_successfully = False

        self.status_update.emit(
            f"Attempting to set montage. Type: '{montage_type}', Identifier: '{montage_identifier}'")

        try:
            if montage_type == 'template':
                montage = mne.channels.make_standard_montage(montage_identifier)
                raw.set_montage(montage, match_case=False, on_missing='warn')
                montage_applied_successfully = True
                self.status_update.emit(f"Applied template montage: {montage_identifier}")

            elif montage_type == 'custom':
                if not os.path.exists(montage_identifier):
                    self.status_update.emit(f"Custom montage file not found: {montage_identifier}")
                    raise ValueError(f"Custom montage file not found: {montage_identifier}")

                file_ext = Path(montage_identifier).suffix.lower()
                ch_pos_dict = None
                montage_created_from_dict = False

                if file_ext == '.mat':
                    self.status_update.emit(f"Reading custom .mat montage: {montage_identifier}")
                    try:
                        ch_pos_dict = CleanEEGWorker._read_mat_locations(montage_identifier)
                    except Exception as e_mat:
                        self.status_update.emit(f"Error reading .mat file '{montage_identifier}': {e_mat}")
                elif file_ext in ['.ced', '.csd']:
                    self.status_update.emit(f"Reading custom {file_ext} montage: {montage_identifier}")
                    try:
                        ch_pos_dict = CleanEEGWorker._read_ced_locations(montage_identifier)
                    except Exception as e_ced:
                        self.status_update.emit(f"Error reading {file_ext} file '{montage_identifier}': {e_ced}")

                if ch_pos_dict:
                    self.status_update.emit(f"Successfully read channel positions from {file_ext} file.")
                    try:
                        # Ensure all channel names are strings for make_dig_montage
                        ch_pos_dict_str_keys = {str(k): v for k, v in ch_pos_dict.items()}
                        montage = mne.channels.make_dig_montage(ch_pos=ch_pos_dict_str_keys, coord_frame='head')
                        raw.set_montage(montage, match_case=False, on_missing='warn')
                        montage_applied_successfully = True
                        montage_created_from_dict = True
                        self.status_update.emit(f"Applied custom montage from {file_ext} file using make_dig_montage.")
                    except Exception as e_make_dig:
                        self.status_update.emit(
                            f"Error creating DigMontage from {file_ext} data: {e_make_dig}. Will attempt fallback.")

                if not montage_created_from_dict:
                    self.status_update.emit(
                        f"Attempting fallback: mne.channels.read_custom_montage for: {montage_identifier}")
                    try:
                        montage = mne.channels.read_custom_montage(montage_identifier)
                        raw.set_montage(montage, match_case=False, on_missing='warn')
                        montage_applied_successfully = True
                        self.status_update.emit(
                            f"Applied custom montage using read_custom_montage for: {montage_identifier}")
                    except Exception as e_read_custom:
                        self.status_update.emit(
                            f"Fallback read_custom_montage also failed for '{montage_identifier}': {e_read_custom}")

            else:
                self.status_update.emit(f"Unknown montage type: '{montage_type}'. Montage not applied.")

        except Exception as e_set_montage_outer:
            self.status_update.emit(
                f"Outer error during montage setting for '{montage_identifier}':"
                f"{e_set_montage_outer}. Montage may not be applied.")

        if not montage_applied_successfully:
            self.status_update.emit(
                f"Warning: Montage '{montage_identifier}'"
                f"(type: {montage_type}) could not be fully applied. Processing continues.")
        elif raw.get_montage() is None:
            self.status_update.emit(
                f"Warning: Montage was reportedly applied,"
                f"but raw.get_montage() is still None for '{montage_identifier}'.")
        else:
            self.status_update.emit(
                f"Montage '{montage_identifier}' (type: {montage_type}) successfully set and verified.")

    @staticmethod
    def _remove_line_noise(raw: mne.io.Raw, line_freq: int) -> mne.io.Raw:
        """Remove line noise using DSS"""
        eeg_data = raw.get_data()
        processed_data, _ = dss.dss_line(
            eeg_data.T,
            fline=line_freq,
            sfreq=raw.info['sfreq'],
            show=False
        )
        raw._data = processed_data.T
        return raw

    @staticmethod
    def _detect_bad_channels(raw: mne.io.Raw) -> List[str]:
        """Detect bad channels using PyPrep"""
        picks_eeg_only = mne.pick_types(raw.info, eeg=True, eog=False)
        nd = NoisyChannels(raw.copy().pick(picks=picks_eeg_only), random_state=1337)
        nd.find_all_bads()

        if nd:
            bad_channels = [str(ch) for ch in nd.get_bads()]
            return bad_channels
        return []

    @staticmethod
    def _apply_ica(raw: mne.io.Raw, settings: Dict[str, Any]) -> Tuple[mne.io.Raw, ICA]:
        """Apply ICA and remove artifacts"""
        ica_method = settings.get('ica_method', 'fastica')
        ica = ICA(n_components=None, random_state=97, method=ica_method)
        ica.fit(raw)

        # Use ICLabel to classify components
        ic_labels = label_components(raw, ica, method='iclabel')
        labels = ic_labels["labels"]

        # Determine which components to exclude based on checked options
        exclude_types = []
        if settings.get('remove_muscle'):
            exclude_types.append('muscle artifact')
        if settings.get('remove_eye_blink'):
            exclude_types.append('eye blink')
        if settings.get('remove_heart_beat'):
            exclude_types.append('heart beat')
        if settings.get('remove_others'):
            exclude_types.extend(['line noise', 'channel noise'])

        exclude_idx = [
            idx for idx, label in enumerate(labels)
            if label in exclude_types
        ]

        ica.exclude = exclude_idx
        return ica.apply(raw), ica

    @staticmethod
    def _apply_asr(raw: mne.io.Raw, cutoff: float, asr_calibration: float) -> mne.io.Raw:
        """Apply ASR; if anything goes wrong, return raw unchanged."""
        picks_eeg_good = mne.pick_types(raw.info, eeg=True, eog=False, exclude='bads')
        eeg_data_for_asr = raw.get_data(picks=picks_eeg_good)

        try:
            asr = ASR(sfreq=raw.info['sfreq'], cutoff=cutoff, method='euclid')

            # Fit ASR on a clean portion of the data
            train_duration = min(asr_calibration, raw.times[-1])
            train_data = eeg_data_for_asr[:, : int(train_duration * raw.info['sfreq'])]
            asr.fit(train_data)

            # Transform the entire dataset
            cleaned_data = asr.transform(eeg_data_for_asr)
            raw._data[picks_eeg_good] = cleaned_data

        except Exception:
            # If any error occurs, return raw without modification
            return raw

        return raw

    def _save_processed_data(self, raw: mne.io.Raw, original_filename: str) -> str:
        """Save processed data, preserving input subfolder structure."""
        input_base_dir_str = self.settings.get('input_base_dir', None)
        original_filepath = Path(self.file_path)

        if input_base_dir_str:
            input_base_dir = Path(input_base_dir_str)
            try:
                relative_dir = original_filepath.parent.relative_to(input_base_dir)
            except ValueError:
                # This can happen if original_filepath.parent is not a subpath of input_base_dir
                # (e.g., if input_base_dir is a file, or they are on different drives)
                # In such cases, save directly to output_path without subfolders.
                relative_dir = Path(".")
            target_output_dir = Path(self.output_path) / relative_dir
        else:
            # If no input_base_dir, save directly to output_path
            target_output_dir = Path(self.output_path)

        target_output_dir.mkdir(parents=True, exist_ok=True)

        base_name = Path(original_filename).stem
        output_format = self.settings.get('output_format', 'auto')
        output_file = ""

        if output_format == 'auto':
            original_ext = Path(self.file_path).suffix.lower()
            if original_ext == '.vhdr':
                output_file = target_output_dir / f"{base_name}_clean.vhdr"
                raw.export(str(output_file), fmt='brainvision', overwrite=True)
            elif original_ext == '.set':
                output_file = target_output_dir / f"{base_name}_clean.set"
                raw.export(str(output_file), fmt='eeglab', overwrite=True)
            else:
                output_file = target_output_dir / f"{base_name}_clean.edf"
                raw.export(str(output_file), fmt='edf', overwrite=True)
        else:
            if output_format == 'brainvision':
                output_file = target_output_dir / f"{base_name}_clean.vhdr"
                raw.export(str(output_file), fmt='brainvision', overwrite=True)
            elif output_format == 'eeglab':
                output_file = target_output_dir / f"{base_name}_clean.set"
                raw.export(str(output_file), fmt='eeglab', overwrite=True)
            elif output_format == 'edf':
                output_file = target_output_dir / f"{base_name}_clean.edf"
                raw.export(str(output_file), fmt='edf', overwrite=True)

        return str(output_file)

    def _generate_enhanced_report(self, raw_orig: mne.io.Raw, raw_processed: mne.io.Raw,
                                  bads_detected: List[str], original_filename: str,
                                  input_base_dir_str: Optional[str] = None):
        """Generates an enhanced HTML report with quality assessment metrics."""
        try:
            self.status_update.emit("Initializing enhanced report...")
            report_title = f"Enhanced Preprocessing Report for {original_filename}"
            report = mne.Report(title=report_title, verbose=False)

            main_section_title = "EEG Preprocessing Pipeline with Quality Assessment"

            # Ensure we have processing stages and SNR data
            if not hasattr(self, 'processing_stages') or not self.processing_stages:
                self.status_update.emit("Warning: No processing stages found, creating minimal report...")
                # Create minimal processing stages
                self.processing_stages = {
                    'Original': raw_orig.copy(),
                    'Final': raw_processed.copy()
                }

            if not hasattr(self, 'snr_log') or not self.snr_log:
                self.status_update.emit("Computing SNR for report...")
                # Compute SNR for available stages
                self.snr_log = {}
                for stage_name, stage_raw in self.processing_stages.items():
                    try:
                        self.snr_log[stage_name] = self.compute_snr(stage_raw)
                    except Exception as e:
                        self.status_update.emit(f"Could not compute SNR for {stage_name}: {e}")
                        # Create dummy SNR data
                        self.snr_log[stage_name] = {
                            'alpha': {'mean_snr': 0.0, 'std_snr': 0.0},
                            'beta': {'mean_snr': 0.0, 'std_snr': 0.0},
                            'theta': {'mean_snr': 0.0, 'std_snr': 0.0},
                            'delta': {'mean_snr': 0.0, 'std_snr': 0.0},
                            'gamma': {'mean_snr': 0.0, 'std_snr': 0.0}
                        }

            # Add comprehensive summary at the beginning
            try:
                summary_html = self._create_summary_html(original_filename)
                report.add_html(summary_html, title="Executive Summary", section=main_section_title, tags=("Summary",))
            except Exception as e:
                self.status_update.emit(f"Could not create summary: {e}")

            # Add SNR improvement table
            try:
                snr_table_html = self._create_snr_summary_table()
                report.add_html(
                    snr_table_html, title="SNR Improvement Summary", section=main_section_title,
                    tags=("SNR", "Summary"))
            except Exception as e:
                self.status_update.emit(f"Could not create SNR table: {e}")

            # Process each stage and add to report
            for stage_name, stage_raw in self.processing_stages.items():
                try:
                    self._add_stage_to_report(report, stage_name, stage_raw, main_section_title)
                except Exception as e:
                    self.status_update.emit(f"Could not add stage {stage_name} to report: {e}")

            # Add overall comparison plots
            try:
                self._add_comparison_plots(report, main_section_title)
            except Exception as e:
                self.status_update.emit(f"Could not add comparison plots: {e}")

            # Save report
            original_filepath = Path(self.file_path)
            if input_base_dir_str:
                input_base_dir = Path(input_base_dir_str)
                try:
                    relative_dir = original_filepath.parent.relative_to(input_base_dir)
                except ValueError:
                    relative_dir = Path(".")
                target_report_dir = Path(self.output_path) / relative_dir
            else:
                target_report_dir = Path(self.output_path)

            target_report_dir.mkdir(parents=True, exist_ok=True)
            report_filename = target_report_dir / f"{Path(original_filename).stem}_enhanced_report.html"

            self.status_update.emit(f"Saving enhanced report to: {str(report_filename)}")
            try:
                report.save(str(report_filename), overwrite=True, open_browser=False)
                self.status_update.emit(f"Enhanced report successfully saved to {report_filename}")
            except Exception as e_save:
                save_err_msg = f"Failed to save report to {report_filename}. Error: {e_save}"
                self.status_update.emit(save_err_msg)
                self.processing_error.emit(original_filename, save_err_msg)

        except Exception as e_generate:
            gen_err_msg = f"Error during report generation for {original_filename}: {e_generate}"
            self.status_update.emit(gen_err_msg)
            self.processing_error.emit(original_filename, gen_err_msg)

    def _create_summary_html(self, filename: str) -> str:
        """Create executive summary HTML"""
        stages = list(self.processing_stages.keys()) if hasattr(self, 'processing_stages') else ['Original', 'Final']

        # Get current date/time properly formatted
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = f"""
        <h2>Executive Summary</h2>
        <p><strong>File:</strong> {filename}</p>
        <p><strong>Processing Date:</strong> {current_time}</p>
        <p><strong>Processing Steps Completed:</strong></p>
        <ol>
        """

        for stage in stages:
            html += f"<li>{stage}</li>"

        html += """
        </ol>
        <p><strong>Key Results:</strong></p>
        <ul>
        """

        # Add key metrics safely
        try:
            if hasattr(self, 'snr_log') and self.snr_log and 'Original' in self.snr_log and 'Final' in self.snr_log:
                orig_alpha = self.snr_log['Original']['alpha']['mean_snr']
                final_alpha = self.snr_log['Final']['alpha']['mean_snr']
                improvement = final_alpha - orig_alpha
                html += f"<li>Alpha band SNR improvement: {improvement:+.2f} dB</li>"
            else:
                html += "<li>SNR data unavailable</li>"

            # Count channels safely
            if hasattr(self, 'processing_stages') and 'Original' in self.processing_stages:
                n_channels = len(self.processing_stages['Original'].ch_names)
                html += f"<li>Total channels processed: {n_channels}</li>"
        except Exception as e:
            html += f"<li>Error computing metrics: {str(e)}</li>"

        html += "</ul>"
        return html

    def _create_snr_summary_table(self) -> str:
        """Create SNR summary table HTML"""
        if not hasattr(self, 'snr_log') or not self.snr_log:
            return "<p>No SNR data available</p>"

        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        stages = list(self.snr_log.keys())

        html = """
        <style>
            .snr-table { border-collapse: collapse; width: 100%; margin: 20px 0; }
            .snr-table th, .snr-table td { border: 1px solid #ddd; padding: 8px; text-align: center; }
            .snr-table th { background-color: #f2f2f2; font-weight: bold; }
            .snr-table tr:nth-child(even) { background-color: #f9f9f9; }
            .improvement { color: green; font-weight: bold; }
            .degradation { color: red; font-weight: bold; }
        </style>
        <h3>Signal-to-Noise Ratio (SNR) by Frequency Band and Processing Stage</h3>
        <table class="snr-table">
        <tr>
            <th>Stage</th>
        """

        # Add column headers for each band
        for band in bands:
            html += f"<th>{band.capitalize()} (dB)</th>"
        html += "</tr>"

        # Add rows for each stage
        for stage in stages:
            html += f"<tr><td><strong>{stage}</strong></td>"
            for band in bands:
                try:
                    if band in self.snr_log[stage]:
                        snr_val = self.snr_log[stage][band]['mean_snr']
                        if isinstance(snr_val, (int, float)) and not np.isnan(snr_val):
                            html += f"<td>{snr_val:.2f}</td>"
                        else:
                            html += "<td>N/A</td>"
                    else:
                        html += "<td>N/A</td>"
                except (KeyError, TypeError, ValueError):
                    html += "<td>N/A</td>"
            html += "</tr>"

        # Add improvement row if we have Original and Final
        if 'Original' in self.snr_log and 'Final' in self.snr_log:
            html += "<tr><td><strong>Total Improvement</strong></td>"
            for band in bands:
                try:
                    if (band in self.snr_log['Original'] and band in self.snr_log['Final']):
                        orig_val = self.snr_log['Original'][band]['mean_snr']
                        final_val = self.snr_log['Final'][band]['mean_snr']

                        if (isinstance(orig_val, (int, float)) and isinstance(final_val, (int, float)) and
                                not np.isnan(orig_val) and not np.isnan(final_val)):
                            improvement = final_val - orig_val
                            css_class = 'improvement' if improvement > 0 else 'degradation' if improvement < 0 else ''
                            html += f'<td class="{css_class}">{improvement:+.2f}</td>'
                        else:
                            html += "<td>N/A</td>"
                    else:
                        html += "<td>N/A</td>"
                except (KeyError, TypeError, ValueError):
                    html += "<td>N/A</td>"
            html += "</tr>"

        html += "</table>"
        return html

    def _add_stage_to_report(self, report: mne.Report, stage_name: str, stage_raw: mne.io.Raw, section_title: str):
        """Add a processing stage to the report with quality metrics"""
        self.status_update.emit(f"Adding {stage_name} to report...")

        # Add stage header
        try:
            report.add_html(
                f"<h2>{stage_name}</h2>", title=f"{stage_name} Header", section=section_title,
                tags=(stage_name, "Header"))
        except Exception as e:
            self.status_update.emit(f"Could not add header for {stage_name}: {e}")

        # Add SNR metrics for this stage
        try:
            if hasattr(self, 'snr_log') and stage_name in self.snr_log:
                snr_html = self._create_stage_snr_html(stage_name)
                report.add_html(
                    snr_html, title=f"{stage_name} SNR Metrics", section=section_title, tags=(stage_name, "SNR"))
        except Exception as e:
            self.status_update.emit(f"Could not add SNR metrics for {stage_name}: {e}")

        # Add PSD plot
        try:
            psd_computed = stage_raw.compute_psd(fmax=80, verbose=False)
            fig_psd = psd_computed.plot(show=False, average=True, spatial_colors=False)
            fig_psd.suptitle(f"PSD: {stage_name}")
            report.add_figure(fig_psd, title=f"PSD: {stage_name}", section=section_title, tags=(stage_name, "PSD"))
            plt.close(fig_psd)
        except Exception as e:
            self.status_update.emit(f"Could not create PSD for {stage_name}: {e}")

        # Add data snippet (simplified to avoid potential issues)
        try:
            # Create a simple data plot instead of using add_raw which can be problematic
            fig, ax = plt.subplots(1, 1, figsize=(12, 6))

            # Plot first 10 seconds of data from first 10 channels
            n_channels = min(10, len(stage_raw.ch_names))
            n_times = min(int(10 * stage_raw.info['sfreq']), stage_raw.n_times)

            data_snippet = stage_raw.get_data()[:n_channels, :n_times]
            times_snippet = stage_raw.times[:n_times]

            for i in range(n_channels):
                ax.plot(times_snippet, data_snippet[i] + i * 100e-6, linewidth=0.5)

            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude (V)')
            ax.set_title(f"Data Sample: {stage_name} (first {n_channels} channels, 10s)")
            ax.grid(True, alpha=0.3)

            report.add_figure(fig, title=f"Data Sample: {stage_name}", section=section_title, tags=(stage_name, "Data"))
            plt.close(fig)

        except Exception as e:
            self.status_update.emit(f"Could not add data snippet for {stage_name}: {e}")

    def _create_stage_snr_html(self, stage_name: str) -> str:
        """Create HTML for SNR metrics of a specific stage"""
        if not hasattr(self, 'snr_log') or stage_name not in self.snr_log:
            return f"<p>No SNR data for {stage_name}</p>"

        snr_data = self.snr_log[stage_name]

        html = f"""
        <h4>SNR Metrics for {stage_name}</h4>
        <table style="border-collapse: collapse; margin: 10px 0;">
        <tr>
            <th style="border: 1px solid #ddd; padding: 5px;">Band</th>
            <th style="border: 1px solid #ddd; padding: 5px;">Mean SNR (dB)</th>
            <th style="border: 1px solid #ddd; padding: 5px;">Std Dev (dB)</th>
        </tr>
        """

        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            try:
                if band in snr_data:
                    mean_snr = snr_data[band]['mean_snr']
                    std_snr = snr_data[band]['std_snr']

                    # Check if values are valid numbers
                    if isinstance(mean_snr, (int, float)) and isinstance(std_snr, (int, float)):
                        if not (np.isnan(mean_snr) or np.isnan(std_snr)):
                            html += f"""
                            <tr>
                                <td style="border: 1px solid #ddd; padding: 5px;">{band.capitalize()}</td>
                                <td style="border: 1px solid #ddd; padding: 5px;">{mean_snr:.2f}</td>
                                <td style="border: 1px solid #ddd; padding: 5px;">{std_snr:.2f}</td>
                            </tr>
                            """
                        else:
                            html += f"""
                            <tr>
                                <td style="border: 1px solid #ddd; padding: 5px;">{band.capitalize()}</td>
                                <td style="border: 1px solid #ddd; padding: 5px;">N/A</td>
                                <td style="border: 1px solid #ddd; padding: 5px;">N/A</td>
                            </tr>
                            """
                    else:
                        html += f"""
                        <tr>
                            <td style="border: 1px solid #ddd; padding: 5px;">{band.capitalize()}</td>
                            <td style="border: 1px solid #ddd; padding: 5px;">N/A</td>
                            <td style="border: 1px solid #ddd; padding: 5px;">N/A</td>
                        </tr>
                        """
            except (KeyError, TypeError, ValueError):
                html += f"""
                <tr>
                    <td style="border: 1px solid #ddd; padding: 5px;">{band.capitalize()}</td>
                    <td style="border: 1px solid #ddd; padding: 5px;">Error</td>
                    <td style="border: 1px solid #ddd; padding: 5px;">Error</td>
                </tr>
                """

        html += "</table>"
        return html

    def _add_comparison_plots(self, report: mne.Report, section_title: str):
        """Add comprehensive comparison plots to the report"""
        self.status_update.emit("Creating comparison plots...")

        # 1. SNR progression plot
        try:
            fig_snr = self._create_snr_progression_plot()
            if fig_snr:
                report.add_figure(
                    fig_snr, title="SNR Progression Across Processing Steps", section=section_title,
                    tags=("Comparison", "SNR"))
                plt.close(fig_snr)
        except Exception as e:
            self.status_update.emit(f"Could not create SNR progression plot: {e}")

        # 2. Before/After PSD comparison
        try:
            fig_psd_comp = self._create_psd_comparison_plot()
            if fig_psd_comp:
                report.add_figure(
                    fig_psd_comp, title="PSD Comparison: Original vs Final", section=section_title,
                    tags=("Comparison", "PSD"))
                plt.close(fig_psd_comp)
        except Exception as e:
            self.status_update.emit(f"Could not create PSD comparison: {e}")

    def _create_snr_progression_plot(self) -> Optional[Figure]:
        """Create SNR progression plot"""
        if not hasattr(self, 'snr_log') or not self.snr_log:
            return None

        try:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()

            bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
            stages = list(self.snr_log.keys())

            for i, band in enumerate(bands):
                if i < len(axes):
                    snr_values = []
                    valid_stages = []

                    for stage in stages:
                        try:
                            if band in self.snr_log[stage]:
                                mean_snr = self.snr_log[stage][band]['mean_snr']
                                if isinstance(mean_snr, (int, float)) and not np.isnan(mean_snr):
                                    snr_values.append(mean_snr)
                                    valid_stages.append(stage)
                        except (KeyError, TypeError):
                            continue

                    if snr_values and len(snr_values) > 1:
                        axes[i].plot(range(len(valid_stages)), snr_values, 'o-', linewidth=2, markersize=8)
                        axes[i].set_title(f'{band.capitalize()} Band SNR')
                        axes[i].set_ylabel('SNR (dB)')
                        axes[i].set_xticks(range(len(valid_stages)))
                        axes[i].set_xticklabels(valid_stages, rotation=45, ha='right')
                        axes[i].grid(True, alpha=0.3)

                        # Add improvement annotation
                        improvement = snr_values[-1] - snr_values[0]
                        color = 'green' if improvement > 0 else 'red'
                        axes[i].text(
                            0.95, 0.95, f'Δ = {improvement:+.1f} dB',
                            transform=axes[i].transAxes,
                            ha='right', va='top',
                            bbox=dict(boxstyle='round', facecolor=color, alpha=0.3))
                    else:
                        axes[i].text(0.5, 0.5, 'No valid data', transform=axes[i].transAxes, ha='center')
                        axes[i].set_title(f'{band.capitalize()} Band SNR')

            # Remove empty subplot
            if len(bands) < len(axes):
                fig.delaxes(axes[-1])

            fig.suptitle('Signal-to-Noise Ratio Progression Throughout Processing', fontsize=16)
            plt.tight_layout()

            return fig

        except Exception as e:
            self.status_update.emit(f"Error creating SNR progression plot: {e}")
            return None

    def _create_psd_comparison_plot(self) -> Optional[Figure]:
        """Create before/after PSD comparison plot"""
        if not hasattr(self, 'processing_stages') or not self.processing_stages:
            return None

        if 'Original' not in self.processing_stages or 'Final' not in self.processing_stages:
            # Use first and last available stages
            stage_names = list(self.processing_stages.keys())
            if len(stage_names) < 2:
                return None
            first_stage = stage_names[0]
            last_stage = stage_names[-1]
        else:
            first_stage = 'Original'
            last_stage = 'Final'

        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

            # Get PSDs
            orig_raw = self.processing_stages[first_stage]
            final_raw = self.processing_stages[last_stage]

            # Plot Original PSD
            psd_orig = orig_raw.compute_psd(fmax=80, verbose=False)
            psd_orig_data = 10 * np.log10(psd_orig.get_data().mean(axis=0) + 1e-12)
            ax1.plot(psd_orig.freqs, psd_orig_data, 'b-', linewidth=2)
            ax1.set_title(f'{first_stage} Data')
            ax1.set_xlabel('Frequency (Hz)')
            ax1.set_ylabel('Power (dB)')
            ax1.grid(True, alpha=0.3)

            # Plot Final PSD
            psd_final = final_raw.compute_psd(fmax=80, verbose=False)
            psd_final_data = 10 * np.log10(psd_final.get_data().mean(axis=0) + 1e-12)
            ax2.plot(psd_final.freqs, psd_final_data, 'r-', linewidth=2)
            ax2.set_title(f'{last_stage} Processed Data')
            ax2.set_xlabel('Frequency (Hz)')
            ax2.set_ylabel('Power (dB)')
            ax2.grid(True, alpha=0.3)

            # Plot comparison
            ax3.plot(psd_orig.freqs, psd_orig_data, 'b-', linewidth=2, label=first_stage, alpha=0.7)
            ax3.plot(psd_final.freqs, psd_final_data, 'r-', linewidth=2, label=last_stage, alpha=0.7)
            ax3.set_title('PSD Comparison')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Power (dB)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)

            # Highlight frequency bands
            for ax in [ax1, ax2, ax3]:
                ax.axvspan(1, 4, alpha=0.1, color='purple')
                ax.axvspan(4, 8, alpha=0.1, color='blue')
                ax.axvspan(8, 13, alpha=0.1, color='green')
                ax.axvspan(13, 30, alpha=0.1, color='orange')
                ax.axvspan(30, 80, alpha=0.1, color='red')

            fig.suptitle('Power Spectral Density: Before and After Processing', fontsize=16)
            plt.tight_layout()

            return fig

        except Exception as e:
            self.status_update.emit(f"Error creating PSD comparison plot: {e}")
            return None


class CleanEEGController(QWidget):
    """Main controller for the EEG processing application"""

    def __init__(self, ui_file: str):
        super().__init__()
        self.ui = uic.loadUi(ui_file, self)
        self.loaded_files = []
        self.current_raw_for_montage = None
        self.processing_thread = None
        self.montage_options = self._get_available_montages()
        self.current_file_index = -1
        self.current_file_path = None

        # Create plot widgets
        self.montage_plot_widget = None

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Initialize UI elements"""

        # Add validators for numeric inputs (before setting defaults)
        freq_validator = QDoubleValidator(0.1, 999.0, 2)
        self.step2_hp_filter_lineedit.setValidator(freq_validator)
        self.step2_lp_filter_lineedit.setValidator(freq_validator)

        # ASR validators
        asr_cutoff_validator = QDoubleValidator(1.0, 99.0, 1)
        self.asr_cutoff_lineedit.setValidator(asr_cutoff_validator)

        asr_calibration_validator = QIntValidator(1, 999)
        self.asr_calibration_lineedit.setValidator(asr_calibration_validator)

        # Integer validator for downsample frequency
        ds_validator = QIntValidator(1, 9999)
        self.step2_downsample_lineedit.setValidator(ds_validator)

        # Initialize preprocessing options based on checkbox states
        self._update_preprocessing_options_state()

        # Populate montage combo box
        self.step2_template_montage_combobox.clear()
        self.step2_template_montage_combobox.addItems(self.montage_options)

        # Set default montage
        default_montage = 'standard_1020'
        if default_montage in self.montage_options:
            idx = self.montage_options.index(default_montage)
            self.step2_template_montage_combobox.setCurrentIndex(idx)

        # Clear log line edit
        self.log_lineedit.clear()

    def _setup_default_restoration(self):
        """Setup default value restoration for line edits"""
        self.line_edit_defaults = {
            self.step2_hp_filter_lineedit: '1',
            self.step2_lp_filter_lineedit: '100',
            self.step2_downsample_lineedit: '500',
            self.asr_cutoff_lineedit: '5',
            self.asr_calibration_lineedit: '60'
        }

        for line_edit, default_value in self.line_edit_defaults.items():
            line_edit.setText(default_value)
            # Connect to both signals for comprehensive coverage
            line_edit.editingFinished.connect(
                lambda checked=False, le=line_edit, dv=default_value:
                self._restore_default_if_empty(le, dv)
            )
            line_edit.textChanged.connect(
                lambda text, le=line_edit, dv=default_value:
                self._check_empty_and_restore(le, dv, text)
            )

    @staticmethod
    def _restore_default_if_empty(line_edit, default_value):
        """Restore default value if line edit is empty"""
        if not line_edit.text().strip():
            line_edit.setText(default_value)

    @staticmethod
    def _check_empty_and_restore(line_edit, default_value, current_text):
        """Immediately restore default when text becomes empty"""
        if not current_text.strip():
            QTimer.singleShot(0, lambda: line_edit.setText(default_value))

    def _get_available_montages(self) -> List[str]:
        """Get list of all available MNE montages"""
        # Get all available standard montages
        montages = []

        # Try to get all montages by checking the montage module
        try:
            # Standard montages available in MNE
            standard_montages = [
                'standard_1005',
                'standard_1020',
                'standard_alphabetic',
                'standard_postfixed',
                'standard_prefixed',
                'standard_primed',
                'biosemi16',
                'biosemi32',
                'biosemi64',
                'biosemi128',
                'biosemi160',
                'biosemi256',
                'easycap-M1',
                'easycap-M10',
                'EGI_256',
                'GSN-HydroCel-32',
                'GSN-HydroCel-64_1.0',
                'GSN-HydroCel-65_1.0',
                'GSN-HydroCel-128',
                'GSN-HydroCel-129',
                'GSN-HydroCel-256',
                'GSN-HydroCel-257',
                'mgh60',
                'mgh70',
                'artinis-octamon',
                'artinis-brite23',
                'brainproducts-RNP-BA-128',
            ]

            # Test each montage to see if it's available
            for montage_name in standard_montages:
                try:
                    mne.channels.make_standard_montage(montage_name)
                    montages.append(montage_name)
                except:
                    pass

        except Exception as e:
            self._log(f"Error getting montages: {str(e)}", is_error=True)
            # Fall back to known working montages
            montages = [
                'standard_1005',
                'standard_1020',
                'biosemi128',
                'biosemi256',
                'GSN-HydroCel-128',
                'GSN-HydroCel-256',
                'easycap-M1',
                'easycap-M10',
            ]

        return sorted(montages)

    def _log(self, message: str, is_error: bool = False):
        """Display log message in the log line edit with appropriate color.
           Note: QLineEdit does not render HTML. This will color the entire text.
        """
        style_sheet = "font-size: 14pt;"
        if is_error:
            style_sheet += " color: red;"
        else:
            style_sheet += " color: green;"
        self.log_lineedit.setStyleSheet(style_sheet)
        self.log_lineedit.setText(message)

    def _connect_signals(self):
        """Connect UI signals to slots"""
        # Preprocess Data tab
        self.step1_input_path_button.clicked.connect(self._select_input_directory)
        self.step1_load_pattern_radio.toggled.connect(self._toggle_pattern_input)
        self.step1_import_raw_button.clicked.connect(self._import_data)
        self.step2_save_path_button.clicked.connect(self._select_output_directory)
        self.step2_preprocess_data_button.clicked.connect(self._start_preprocessing)

        # Connect preprocessing option checkboxes to enable/disable their inputs
        self.step2_bp_filter_checkbox.toggled.connect(self._update_preprocessing_options_state)
        self.step2_downsample_checkbox.toggled.connect(self._update_preprocessing_options_state)
        self.step2_line_noise_checkbox.toggled.connect(self._update_preprocessing_options_state)
        self.step2_ica_checkbox.toggled.connect(self._update_preprocessing_options_state)
        self.step2_asr_checkbox.toggled.connect(self._update_preprocessing_options_state)
        self._setup_default_restoration()

        # Connect montage selection change
        self.step2_template_montage_combobox.currentTextChanged.connect(self._on_montage_selection_changed_for_plot)

        # File list management
        self.loaded_selected_files_list.itemSelectionChanged.connect(self._on_file_selection_changed)
        self.loaded_remove_file_button.clicked.connect(self._remove_selected_file)
        self.loaded_clear_files_button.clicked.connect(self._clear_all_files)

    def _update_preprocessing_options_state(self):
        """Enable/disable preprocessing options based on checkbox states"""
        # Band-pass filter
        self.step2_hp_filter_lineedit.setEnabled(self.step2_bp_filter_checkbox.isChecked())
        self.step2_lp_filter_lineedit.setEnabled(self.step2_bp_filter_checkbox.isChecked())

        # Downsample
        self.step2_downsample_lineedit.setEnabled(self.step2_downsample_checkbox.isChecked())

        # Line noise
        line_noise_enabled = self.step2_line_noise_checkbox.isChecked()
        self.line_50_radio.setEnabled(line_noise_enabled)
        self.line_60_radio.setEnabled(line_noise_enabled)

        # ICA options
        ica_enabled = self.step2_ica_checkbox.isChecked()
        self.ica_method_combobox.setEnabled(ica_enabled)
        self.muscle_checkbox.setEnabled(ica_enabled)
        self.eye_blink_checkbox.setEnabled(ica_enabled)
        self.heart_beat_checkbox.setEnabled(ica_enabled)
        self.others_checkbox.setEnabled(ica_enabled)

        # ASR options
        asr_enabled = self.step2_asr_checkbox.isChecked()
        self.asr_calibration_lineedit.setEnabled(asr_enabled)
        self.asr_calibration_label.setEnabled(asr_enabled)
        self.asr_cutoff_lineedit.setEnabled(asr_enabled)
        self.asr_cutoff_label.setEnabled(asr_enabled)

    @pyqtSlot()
    def _select_input_directory(self):
        """Select input directory for EEG files"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select EEG Data Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.step1_input_path_lineedit.setText(directory)
            self.step1_import_raw_button.setEnabled(True)

    @pyqtSlot(bool)
    def _toggle_pattern_input(self, checked: bool):
        """Enable/disable pattern input based on radio button"""
        self.step1_import_pattern_lineedit.setEnabled(checked)

    @pyqtSlot()
    def _import_data(self):
        """Import EEG data files"""
        input_dir = self.step1_input_path_lineedit.text()
        if not input_dir:
            QMessageBox.warning(self, "Warning", "Please select an input directory first.")
            return

        # Get file extension
        format_idx = self.step1_import_format_combobox.currentIndex()
        extensions = self._get_file_extensions(format_idx)

        # Get pattern if specified
        pattern = ""
        if self.step1_load_pattern_radio.isChecked():
            pattern = self.step1_import_pattern_lineedit.text()

        # Find files
        files = self._find_eeg_files(input_dir, extensions, pattern)

        if not files:
            message = "No EEG files found matching the criteria."
            self._log(message)
            QMessageBox.information(self, "Info", message)
            return

        # Add files to list
        self.loaded_files.extend(files)
        self._update_file_list()

        # Enable preprocessing if we have files
        if self.loaded_files:
            self.loaded_remove_file_button.setEnabled(True)
            self.loaded_clear_files_button.setEnabled(True)

        message = f"Loaded {len(files)} EEG files."
        self._log(message)

    @staticmethod
    def _get_file_extensions(format_idx: int) -> List[str]:
        """Get file extensions based on selected format"""
        extension_map = {
            0: ['.vhdr', '.set', '.edf', '.bdf', '.gdf', '.cnt', '.egi', '.mff',
                '.nxe', '.eeg', '.dat', '.fif', '.fif.gz', '.raw', '.raw.fif',
                '.raw.fif.gz'],         # Auto - all supported formats
            1: ['.set'],                # EEGLAB
            2: ['.vhdr'],               # BrainVision
            3: ['.edf'],                # EDF
            4: ['.bdf'],                # BDF
            5: ['.gdf'],                # GDF
            6: ['.cnt'],                # CNT
            7: ['.egi'],                # EGI
            8: ['.mff'],                # MFF
            9: ['.data'],               # Nicolet
            10: ['.nxe'],               # eXimia
            11: ['.lay', '.dat'],       # Persyst
            12: ['.fif', '.fif.gz'],    # FIF
        }
        return extension_map.get(format_idx, [])

    @staticmethod
    def _find_eeg_files(directory: str, extensions: List[str], pattern: str = "") -> List[str]:
        """Find EEG files in directory, avoiding duplicates from same dataset"""
        from fnmatch import fnmatch
        import os

        list_path = []

        # Convert extensions list to handle "auto" case
        if '.auto' in extensions or len(extensions) > 5:  # Auto mode has many extensions
            valid_eeg_formats = [
                ".vhdr", ".edf", ".bdf", ".gdf", ".cnt", ".egi", ".mff", ".set",
                ".data", ".nxe", ".lay", ".dat", ".fif", ".fif.gz", ".raw",
                ".raw.fif", ".raw.fif.gz"
            ]
            search_extensions = valid_eeg_formats
        else:
            search_extensions = extensions

        # Track processed file stems to avoid BrainVision duplicates
        processed_stems = set()

        # Search for files
        for path, subdirs, files in os.walk(directory):
            for name in files:
                file_ext = os.path.splitext(name)[1].lower()
                file_stem = os.path.splitext(name)[0]

                # Check if file matches our criteria
                if file_ext in search_extensions:
                    # Apply pattern matching
                    search_pattern = f"*{pattern}*{file_ext}" if pattern else f"*{file_ext}"
                    if fnmatch(name, search_pattern):

                        # Skip if we've already processed this dataset
                        if file_stem in processed_stems:
                            continue

                        file_path = os.path.join(path, name)

                        # Test if file can be loaded by MNE
                        try:
                            mne.io.read_raw(file_path, preload=False, verbose='ERROR')
                            list_path.append(file_path)
                            processed_stems.add(file_stem)
                        except Exception:
                            # Skip files that can't be loaded
                            continue

        return list_path

    def _update_file_list(self):
        """Update the file list widget"""
        self.loaded_selected_files_list.clear()
        for file in self.loaded_files:
            item = QListWidgetItem(Path(file).name)
            item.setToolTip(file)
            self.loaded_selected_files_list.addItem(item)

    def _on_file_selection_changed(self):
        """Handle file selection change: load info and display montage."""
        selected_items = self.loaded_selected_files_list.selectedItems()
        if selected_items:
            idx = self.loaded_selected_files_list.row(selected_items[0])
            self.current_file_path = self.loaded_files[idx]
            self.current_file_index = idx

            # Log the selection and trigger montage display
            self._log(f"Selected: {Path(self.current_file_path).name}. Displaying montage...")
            self._load_info_and_display_montage(self.current_file_path)

            # Update filename display safely (if the widget still exists for other purposes)
            if hasattr(self, 'vis_figure_compare_filename_lineedit'):
                self.vis_figure_compare_filename_lineedit.setText(Path(self.current_file_path).name)
        else:
            # No file selected, clear montage and related info
            self.current_file_path = None
            self.current_file_index = -1
            self.current_raw_for_montage = None
            self._create_montage_plot()
            self._log("File selection cleared. Montage view updated.")
            if hasattr(self, 'vis_figure_compare_filename_lineedit'):
                self.vis_figure_compare_filename_lineedit.clear()

    def _load_raw_info_only(self, file_path: str) -> Optional[mne.io.Raw]:
        """Load only the header/info of an EEG file without preloading data."""
        try:
            self._log(f"Loading info from: {Path(file_path).name}...")

            # Use mne.io.read_raw which automatically detects the file format
            raw = mne.io.read_raw(file_path, preload=False, verbose='WARNING')

            self._log(f"Info loaded successfully for {Path(file_path).name}.")
            return raw

        except Exception as e:
            error_msg = f"Failed to load info from {Path(file_path).name}: {str(e)}"
            self._log(error_msg, is_error=True)
            return None

    def _load_info_and_display_montage(self, file_path: str):
        """Loads file info and displays the montage."""
        self._log(f"Loading info for {Path(file_path).name} to display montage...")
        raw_info_obj = self._load_raw_info_only(file_path)

        if raw_info_obj:
            self.current_raw_for_montage = raw_info_obj
            self._create_montage_plot()
        else:
            self.current_raw_for_montage = None
            # Clear montage plot area if loading failed
            for i in reversed(range(self.Figure_Layout_Montage.count())):
                widget = self.Figure_Layout_Montage.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            if self.montage_plot_widget:
                self.montage_plot_widget.clear()

            self._log(f"Failed to load info for {Path(file_path).name}, montage view cleared.", is_error=True)

    def _create_montage_plot(self):
        """Create montage plot"""
        if not self.current_raw_for_montage:
            # self._log("No file selected or info loaded for montage.") # Avoid repetitive logs if called often
            # Clear plot area if no data
            for i in reversed(range(self.Figure_Layout_Montage.count())):
                widget = self.Figure_Layout_Montage.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            if self.montage_plot_widget:
                self.montage_plot_widget.clear()
            return

        # Clear existing plot from layout before adding a new one
        for i in reversed(range(self.Figure_Layout_Montage.count())):
            widget = self.Figure_Layout_Montage.itemAt(i).widget()
            if widget:  # Check if widget is not None before calling setParent(None)
                widget.setParent(None)

        # Create plot widget if it doesn't exist or recreate it to ensure proper initialization
        if self.montage_plot_widget:
            self.montage_plot_widget.setParent(None)
            self.montage_plot_widget = None

        self.montage_plot_widget = MNEPlotWidget(parent=self)  # Create fresh widget

        # Get selected montage
        montage_name = self.step2_template_montage_combobox.currentText()

        try:
            self._log(f"Setting montage: {montage_name} for display.")

            raw_for_plot = self.current_raw_for_montage.copy()
            montage = mne.channels.make_standard_montage(montage_name)
            raw_for_plot.set_montage(montage, match_case=False, on_missing='warn')

            # Create a fresh figure and axis
            self.montage_plot_widget.figure.clear()
            ax = self.montage_plot_widget.figure.add_subplot(111)

            # Plot montage
            raw_for_plot.plot_sensors(show_names=True,
                                      show=False, axes=ax, kind='topomap')  # Added kind for better display
            ax.set_title(f"Montage: {montage_name}")

            # Ensure the figure has proper layout
            self.montage_plot_widget.figure.tight_layout()
            self.montage_plot_widget.canvas.draw()

            self.Figure_Layout_Montage.addWidget(self.montage_plot_widget)
            self._log(f"Montage plot displayed using {montage_name}.")

        except Exception as e:
            warning_msg = f"Could not set or plot montage {montage_name}: {str(e)}"
            self._log(warning_msg, is_error=True)
            # QMessageBox.warning(self, "Montage Warning", warning_msg) # Can be too intrusive

    @pyqtSlot()
    def _select_output_directory(self):
        """Select output directory for processed files"""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Output Directory",
            "",
            QFileDialog.ShowDirsOnly
        )
        if directory:
            self.step2_save_path_lineedit.setText(directory)

    def _validate_preprocessing_inputs(self) -> bool:
        """Validate preprocessing input fields"""
        try:
            # Validate bandpass filter frequencies
            if self.step2_bp_filter_checkbox.isChecked():
                hp_freq = float(self.step2_hp_filter_lineedit.text())
                lp_freq = float(self.step2_lp_filter_lineedit.text())
                if hp_freq <= 0 or lp_freq <= 0:
                    raise ValueError("Filter frequencies must be positive")
                if hp_freq >= lp_freq:
                    raise ValueError("Highpass frequency must be less than lowpass frequency")

            # Validate downsample frequency
            if self.step2_downsample_checkbox.isChecked():
                ds_freq = int(self.step2_downsample_lineedit.text())
                if ds_freq <= 0:
                    raise ValueError("Downsample frequency must be positive")

            return True

        except ValueError as e:
            QMessageBox.warning(self, "Invalid Input", str(e))
            return False

    @pyqtSlot()
    def _start_preprocessing(self):
        """Start preprocessing all loaded files"""
        if not self.loaded_files:
            warning_msg = "No files loaded for preprocessing."
            self._log(warning_msg)
            QMessageBox.warning(self, "Warning", warning_msg)
            return

        output_dir = self.step2_save_path_lineedit.text()
        if not output_dir:
            warning_msg = "Please select an output directory."
            self._log(warning_msg)
            QMessageBox.warning(self, "Warning", warning_msg)
            return

        # Validate inputs
        if not self._validate_preprocessing_inputs():
            return

        # Gather preprocessing settings
        settings = self._get_preprocessing_settings()

        # Reset progress and status
        self.preprocess_progressbar.setValue(0)
        self._log(f"Starting preprocessing of {len(self.loaded_files)} files...")

        # Disable UI during processing
        self.step2_preprocess_data_button.setEnabled(False)

        # Process first file (extend to process all files in sequence)
        self._process_next_file(0, output_dir, settings)

    def _get_preprocessing_settings(self) -> Dict[str, Any]:
        """Gather all preprocessing settings from UI"""
        settings = {
            'set_montage': True,  # Always try to set montage
            'apply_bandpass': self.step2_bp_filter_checkbox.isChecked(),
            'highpass_freq':
                float(self.step2_hp_filter_lineedit.text()) if self.step2_bp_filter_checkbox.isChecked() else None,
            'lowpass_freq':
                float(self.step2_lp_filter_lineedit.text()) if self.step2_bp_filter_checkbox.isChecked() else None,
            'apply_downsample': self.step2_downsample_checkbox.isChecked(),
            'resample_freq':
                int(self.step2_downsample_lineedit.text()) if self.step2_downsample_checkbox.isChecked() else None,
            'remove_line_noise': self.step2_line_noise_checkbox.isChecked(),
            'line_freq': 50 if self.line_50_radio.isChecked() else 60,
            'detect_bad_channels': self.step2_prep_checkbox.isChecked(),
            'apply_ica': self.step2_ica_checkbox.isChecked(),
            'ica_method': self.ica_method_combobox.currentText() if self.step2_ica_checkbox.isChecked() else 'fastica',
            'remove_muscle': self.muscle_checkbox.isChecked() if self.step2_ica_checkbox.isChecked() else False,
            'remove_eye_blink': self.eye_blink_checkbox.isChecked() if self.step2_ica_checkbox.isChecked() else False,
            'remove_heart_beat': self.heart_beat_checkbox.isChecked() if self.step2_ica_checkbox.isChecked() else False,
            'remove_others': self.others_checkbox.isChecked() if self.step2_ica_checkbox.isChecked() else False,
            'apply_asr': self.step2_asr_checkbox.isChecked(),
            'asr_cutoff': float(self.asr_cutoff_lineedit.text()) if self.step2_asr_checkbox.isChecked() else 5,
            'asr_calibration':
                float(self.asr_calibration_lineedit.text()) if self.step2_asr_checkbox.isChecked() else 60,
            'interpolate_bads': self.step2_interpolation_checkbox.isChecked(),
            'output_format': self._get_output_format(),
            'export_report':
                self.export_report_checkbox.isChecked() if hasattr(self, 'export_report_checkbox') else False,
            'input_base_dir': self.step1_input_path_lineedit.text()
        }

        # Add montage settings - simplified to always use template montage
        settings['montage'] = self.step2_template_montage_combobox.currentText()
        settings['montage_type'] = 'template'

        return settings

    def _get_output_format(self) -> str:
        """Get output format from UI"""
        format_idx = self.step1_export_format_combobox.currentIndex()
        format_map = {
            0: 'auto',
            1: 'brainvision',
            2: 'eeglab',
            3: 'edf'
        }
        return format_map.get(format_idx, 'auto')

    def _process_next_file(self, file_idx: int, output_dir: str, settings: Dict[str, Any]):
        """Process the next file in the queue"""
        if file_idx >= len(self.loaded_files):
            # All files processed
            self.preprocess_progressbar.setValue(100)
            self.step2_preprocess_data_button.setEnabled(True)
            message = "All files processed successfully!"
            self._log(message)
            QMessageBox.information(self, "Success", message)
            return

        # Update overall progress
        overall_progress = int((file_idx / len(self.loaded_files)) * 100)
        self.preprocess_progressbar.setValue(overall_progress)

        # Create worker thread
        file_path = self.loaded_files[file_idx]
        self.processing_thread = CleanEEGWorker(file_path, output_dir, settings)

        # Connect signals
        self.processing_thread.status_update.connect(
            lambda msg: self._log(f"File {file_idx + 1}/{len(self.loaded_files)}: {msg}")
        )
        self.processing_thread.processing_complete.connect(
            lambda fname, raw: self._on_file_processed(file_idx, output_dir, settings, fname, raw)
        )
        self.processing_thread.processing_error.connect(
            lambda fname, err: self._on_processing_error(fname, err)
        )

        # Start processing
        self.processing_thread.start()

    @pyqtSlot(str, object)
    def _on_file_processed(self, file_idx: int, output_dir: str, settings: Dict[str, Any],
                           filename: str, processed_raw: mne.io.Raw):
        """Handle successful file processing"""
        self._log(f"Successfully processed: {filename}")

        # Store processed data for comparison if it's the currently selected file
        if file_idx == self.current_file_index:
            # self.processed_raw = processed_raw # This attribute is being removed
            # Update filename display (if it exists)
            if hasattr(self, 'vis_figure_compare_filename_lineedit'):
                self.vis_figure_compare_filename_lineedit.setText(filename)
            # Switch to Data Inspection tab - this behavior might need review if tab content changed significantly
            self.new_study_tab_widget.setCurrentIndex(1)

        # Process next file
        self._process_next_file(file_idx + 1, output_dir, settings)

    @pyqtSlot(str, str)
    def _on_processing_error(self, filename: str, error_msg: str):
        """Handle processing error"""
        full_error = f"Error processing {filename}: {error_msg}"
        self._log(full_error, is_error=True)
        QMessageBox.critical(self, "Processing Error", full_error)
        self.step2_preprocess_data_button.setEnabled(True)
        self.preprocess_progressbar.setValue(0)

    @pyqtSlot(str)
    def _on_montage_selection_changed_for_plot(self, montage_name: str):
        """Handle montage selection change for the plot."""
        if self.current_raw_for_montage:
            self._log(f"Montage selection changed to: {montage_name}. Replotting.")
            # self.current_raw_for_montage.set_montage(None) # Not needed as we use a copy in _create_montage_plot
            self._create_montage_plot()

    @pyqtSlot()
    def _remove_selected_file(self):
        """Remove selected file from list"""
        selected_items = self.loaded_selected_files_list.selectedItems()
        if selected_items:
            idx = self.loaded_selected_files_list.row(selected_items[0])
            if 0 <= idx < len(self.loaded_files):
                removed_file = self.loaded_files.pop(idx)
                self._log(f"Removed file: {Path(removed_file).name}")
                self._update_file_list()
                # If the removed file was the currently selected one, clear the montage
                if self.current_file_path == removed_file:
                    self.current_file_path = None
                    self.current_file_index = -1
                    self.current_raw_for_montage = None
                    self._create_montage_plot()
                    if hasattr(self, 'vis_figure_compare_filename_lineedit'):
                        self.vis_figure_compare_filename_lineedit.clear()
            else:
                self._log("Error: Could not remove selected file, index out of range.", is_error=True)
        else:
            self._log("No file selected to remove.", is_error=True)

    @pyqtSlot()
    def _clear_all_files(self):
        """Clear all loaded files"""
        self.loaded_files.clear()
        self._update_file_list()
        self.current_file_path = None
        self.current_file_index = -1
        self.current_raw_for_montage = None
        self._create_montage_plot()
        if hasattr(self, 'vis_figure_compare_filename_lineedit'):
            self.vis_figure_compare_filename_lineedit.clear()


def main():
    app = QApplication(sys.argv)

    # Assuming the UI file is in the same directory
    ui_file = "cleaneeg_interface.ui"

    controller = CleanEEGController(ui_file)
    controller.setWindowTitle("CleanEEG GUI - Enhanced Edition")
    controller.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
