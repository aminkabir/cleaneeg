"""
CleanEEG - Automated Python-based Resting-State EEG Preprocessing GUI

Author: Amin Kabir
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Dict, Any
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

    def run(self):
        try:
            filename = Path(self.file_path).name
            self.status_update.emit(f"Processing {filename}...")

            # Load the data
            self.progress_update.emit(5)
            raw_eeg = self._load_eeg_data(self.file_path)
            raw_original_for_report = raw_eeg.copy()
            bads_before_processing = []
            
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

            # Resample (only if checkbox is checked)
            if self.settings.get('apply_downsample') and self.settings.get('resample_freq'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Downsampling data...")
                raw_eeg.resample(self.settings['resample_freq'])

            raw_clean = raw_eeg.copy()

            # Line noise removal (only if checkbox is checked)
            if self.settings.get('remove_line_noise'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Removing line noise...")
                raw_clean = self._remove_line_noise(raw_clean, self.settings['line_freq'])

            # High-pass filter (only if checkbox is checked)
            if self.settings.get('apply_bandpass') and self.settings.get('highpass_freq'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Applying high-pass filter...")
                raw_clean.filter(l_freq=self.settings['highpass_freq'], h_freq=None, method='fir')

            # Bad channel detection (only if checkbox is checked)
            if self.settings.get('detect_bad_channels'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Detecting bad channels...")
                bad_channels = self._detect_bad_channels(raw_clean)
                raw_clean.info['bads'] = bad_channels
                bads_before_processing = list(bad_channels)

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

            # ASR (only if checkbox is checked)
            if self.settings.get('apply_asr'):
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Applying ASR...")
                asr_cutoff = self.settings.get('asr_cutoff', 5)
                raw_clean = self._apply_asr(raw_clean, asr_cutoff)

            # Interpolate bad channels (only if checkbox is checked and bad channels exist)
            if self.settings.get('interpolate_bads') and len(raw_clean.info['bads']) > 0:
                current_step += 1
                progress = int((current_step / total_steps) * 95)
                self.progress_update.emit(progress)
                self.status_update.emit("Interpolating bad channels...")
                raw_clean.interpolate_bads(reset_bads=True)

            # Save the processed data
            self.progress_update.emit(95)
            self.status_update.emit("Saving processed data...")
            output_file = self._save_processed_data(raw_clean, filename)

            # Generate and save report if requested
            if self.settings.get('export_report'):
                self.status_update.emit("Generating HTML report...")
                self._generate_and_save_report(raw_original_for_report, raw_clean, 
                                               bads_before_processing, filename, 
                                               self.settings.get('input_base_dir'))
                self.status_update.emit("HTML report generated.")

            self.progress_update.emit(100)
            self.processing_complete.emit(filename, raw_clean)

        except Exception as e:
            self.processing_error.emit(filename, str(e))

    def _load_eeg_data(self, file_path: str) -> mne.io.Raw:
        """Load EEG data using automatic format detection"""
        try:
            # Use mne.io.read_raw which automatically detects the file format
            return mne.io.read_raw(file_path, preload=True, verbose='WARNING')
        except Exception as e:
            # If automatic detection fails, provide more specific error message
            ext = Path(file_path).suffix.lower()
            raise ValueError(f"Failed to load file '{file_path}' with extension '{ext}': {str(e)}")

    def _set_montage(self, raw: mne.io.Raw, montage_info: str):
        """Set EEG montage"""
        montage_type = self.settings.get('montage_type', 'template')
        montage_identifier = montage_info
        montage_applied_successfully = False

        self.status_update.emit(f"Attempting to set montage. Type: '{montage_type}', Identifier: '{montage_identifier}'")

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
            
            else: # Unknown montage type
                self.status_update.emit(f"Unknown montage type: '{montage_type}'. Montage not applied.")

        except Exception as e_set_montage_outer:
            self.status_update.emit(
                f"Outer error during montage setting for '{montage_identifier}': {e_set_montage_outer}. Montage may not be applied.")

        if not montage_applied_successfully:
             self.status_update.emit(
                 f"Warning: Montage '{montage_identifier}' (type: {montage_type}) could not be fully applied. Processing continues.")
        elif raw.get_montage() is None:
             self.status_update.emit(
                 f"Warning: Montage was reportedly applied, but raw.get_montage() is still None for '{montage_identifier}'.")
        else:
             self.status_update.emit(
                 f"Montage '{montage_identifier}' (type: {montage_type}) successfully set and verified.")

    def _remove_line_noise(self, raw: mne.io.Raw, line_freq: int) -> mne.io.Raw:
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

    def _detect_bad_channels(self, raw: mne.io.Raw) -> List[str]:
        """Detect bad channels using PyPrep"""
        picks_eeg_only = mne.pick_types(raw.info, eeg=True, eog=False)
        nd = NoisyChannels(raw.copy().pick(picks=picks_eeg_only), random_state=1337)
        nd.find_all_bads()

        if nd:
            bad_channels = [str(ch) for ch in nd.get_bads()]
            return bad_channels
        return []

    def _apply_ica(self, raw: mne.io.Raw, settings: Dict[str, Any]) -> mne.io.Raw:
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
        return ica.apply(raw), ica # Return ICA object as well

    def _apply_asr(self, raw: mne.io.Raw, cutoff: float) -> mne.io.Raw:
        """Apply ASR"""
        picks_eeg_good = mne.pick_types(raw.info, eeg=True, eog=False, exclude='bads')
        eeg_data_for_asr = raw.get_data(picks=picks_eeg_good)

        asr = ASR(sfreq=raw.info['sfreq'], cutoff=cutoff)

        # Fit ASR on a clean portion of the data (first 30 seconds or less)
        train_duration = min(30, raw.times[-1])
        train_data = eeg_data_for_asr[:, :int(train_duration * raw.info['sfreq'])]
        asr.fit(train_data)

        # Transform the entire dataset
        cleaned_data = asr.transform(eeg_data_for_asr)
        raw._data[picks_eeg_good] = cleaned_data

        return raw

    def _save_processed_data(self, raw: mne.io.Raw, original_filename: str) -> str:
        """Save processed data, preserving input subfolder structure."""
        input_base_dir_str = self.settings.get('input_base_dir', None)
        original_filepath = Path(self.file_path) # self.file_path is the full original path

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

    def _generate_and_save_report(self, raw_orig: mne.io.Raw, raw_processed: mne.io.Raw,
                                  bads_detected: List[str], original_filename: str,
                                  input_base_dir_str: Optional[str] = None):
        """Generates and saves an HTML report of the preprocessing steps in a single section."""
        report_generation_successful = False
        try:
            self.status_update.emit("Initializing report...")
            report_title = f"Preprocessing Report for {original_filename}"
            report = mne.Report(title=report_title, verbose=False)

            main_section_title = "EEG Preprocessing Pipeline"
            report.add_html(
                f"<h1>{main_section_title}</h1>", title="Pipeline Header", section=main_section_title,
                tags=("Pipeline", "Header"))

            step_counter = 0
            temp_raw_for_psd = raw_orig.copy()

            # --- Step 0: Original Data ---
            step_counter += 1
            report.add_html(
                f"<h2>Step {step_counter}: Original Data</h2>", title=f"Step{step_counter}_OriginalDataHeader",
                section=main_section_title, tags=("OriginalData", "Header"))
            if raw_orig.get_montage():
                fig_sensors_orig = raw_orig.plot_sensors(show_names=True, show=False)
                report.add_figure(fig_sensors_orig, title="Original Sensor Locations", section=main_section_title,
                                  tags=("OriginalData", "montage"))
                plt.close(fig_sensors_orig)
            report.add_raw(raw_orig, title="Original Raw Data Snippet", psd=False, tags=("OriginalData", "raw_snippet"))
            self._add_psd_plot_to_report(
                report, temp_raw_for_psd, "PSD: Original Data", "OriginalDataPSD",
                main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Step: Montage Setting (if applied) ---
            if self.settings.get('set_montage') and self.settings.get('montage'):
                step_counter += 1
                montage_name = self.settings['montage']
                report.add_html(
                    f"<h2>Step {step_counter}: Set Montage</h2>"
                    f"<p>Attempting to apply montage: {montage_name} to the data for consistent PSD reporting. "
                    f"Effects on data (if any beyond channel locations) will be seen in subsequent PSDs.</p>",
                    title=f"Step{step_counter}_SetMontage", section=main_section_title, tags=("SetMontage", "info"))
                try:
                    montage_obj = mne.channels.make_standard_montage(montage_name)
                    temp_raw_for_psd.set_montage(montage_obj, match_case=False, on_missing='warn')
                    self.status_update.emit(f"Montage '{montage_name}' set for report's temporary raw data.")
                    
                    if temp_raw_for_psd.get_montage():
                        fig_sensors_temp = temp_raw_for_psd.plot_sensors(show_names=True, show=False)
                        report.add_figure(
                            fig_sensors_temp, title=f"Sensor Locations on Temp Data after setting {montage_name}",
                            section=main_section_title, tags=("SetMontage", "temp_montage_plot"))
                        plt.close(fig_sensors_temp)

                except Exception as e_montage:
                    err_msg = f"Could not apply montage '{montage_name}' to temporary data for report: {e_montage}"
                    self.status_update.emit(err_msg)
                    report.add_html(
                        f"<p style=\'color:red;\'>{err_msg}</p>", title="SetMontageError_Report",
                        section=main_section_title, tags=("SetMontage", "error"))
                
                self._add_psd_plot_to_report(
                    report, temp_raw_for_psd, f"PSD: After Attempting to Set Montage ({montage_name})",
                    f"SetMontagePSD", main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Step: Resampling (if applied) ---
            if self.settings.get('apply_downsample') and self.settings.get('resample_freq'):
                step_counter += 1
                resample_f = self.settings['resample_freq']
                report.add_html(f"<h2>Step {step_counter}: Resample Data</h2>"
                                f"<p>Data was resampled to {resample_f} Hz.</p>",
                                title=f"Step{step_counter}_Resample", section=main_section_title,
                                tags=("Resample", "info"))
                temp_raw_for_psd.resample(resample_f)
                self._add_psd_plot_to_report(
                    report, temp_raw_for_psd, f"PSD: After Resampling to {resample_f} Hz", f"ResamplePSD",
                    main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Step: Line Noise Removal (if applied) ---
            if self.settings.get('remove_line_noise'):
                step_counter += 1
                line_f = self.settings['line_freq']
                report.add_html(f"<h2>Step {step_counter}: Line Noise Removal (DSS)</h2>"
                                f"<p>Line noise at {line_f} Hz was removed using DSS.</p>",
                                title=f"Step{step_counter}_LineNoise", section=main_section_title,
                                tags=("LineNoise", "info"))
                
                # Re-apply DSS for PSD on a copy - this is computationally intensive for report
                # Alternative: use a snapshot of raw_clean from worker if available at this stage
                # For now, we'll use the temp_raw_for_psd which has accumulated changes
                data_for_dss = temp_raw_for_psd.get_data()
                data_after_dss, _ = dss.dss_line(
                    data_for_dss.T, fline=line_f, sfreq=temp_raw_for_psd.info['sfreq'], show=False)
                temp_raw_for_psd._data = data_after_dss.T
                self._add_psd_plot_to_report(
                    report, temp_raw_for_psd, f"PSD: After DSS Line Noise ({line_f} Hz) Removal",
                    f"LineNoisePSD", main_section_title, status_update_prefix=f"Step {step_counter}: ")
            
            # --- Step: High-pass Filter (if applied) ---
            if self.settings.get('apply_bandpass') and self.settings.get('highpass_freq'):
                step_counter += 1
                hp_f = self.settings['highpass_freq']
                report.add_html(f"<h2>Step {step_counter}: High-Pass Filter</h2>"
                                f"<p>Applied a high-pass filter at {hp_f} Hz.</p>",
                                title=f"Step{step_counter}_HighPass", section=main_section_title,
                                tags=("HighPass", "info"))
                temp_raw_for_psd.filter(
                    l_freq=hp_f, h_freq=None, method='fir', fir_window='hamming', fir_design='firwin')
                self._add_psd_plot_to_report(
                    report, temp_raw_for_psd, f"PSD: After High-Pass Filter ({hp_f} Hz)", f"HighPassPSD",
                    main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Step: Bad Channel Detection (if applied) ---
            if self.settings.get('detect_bad_channels'):
                step_counter += 1
                report.add_html(
                    f"<h2>Step {step_counter}: Bad Channel Detection (PyPrep)</h2>",
                    title=f"Step{step_counter}_BadChannelsHeader", section=main_section_title,
                    tags=("BadChannels", "Header"))
                if bads_detected:
                    report.add_html(
                        f"<p>Detected bad channels (PyPrep): {', '.join(bads_detected)}</p>",
                        title="Detected Bad Channels List", section=main_section_title,
                        tags=("BadChannels", "PyPrepList"))
                    
                    # Visualize raw data with bad channels marked from original data
                    temp_raw_with_bads_viz = raw_orig.copy()
                    temp_raw_with_bads_viz.info['bads'] = bads_detected
                    if temp_raw_with_bads_viz.get_montage():
                        fig_bads_sensors = temp_raw_with_bads_viz.plot_sensors(show_names=True, show=False)
                        report.add_figure(
                            fig_bads_sensors, title="Sensor Locations with Bad Channels Marked",
                            section=main_section_title, tags=("BadChannels", "BadSensors"))
                        plt.close(fig_bads_sensors)
                    
                    fig_bads_plot = temp_raw_with_bads_viz.plot(
                        n_channels=min(20, len(raw_orig.ch_names)), duration=10, show=False, scalings='auto')
                    report.add_figure(
                        fig_bads_plot, title="Original Data Snippet with Bad Channels Marked",
                        section=main_section_title, tags=("BadChannels", "BadTimeseries"))
                    plt.close(fig_bads_plot)
                else:
                    report.add_html(
                        "<p>No bad channels detected by PyPrep.</p>", title="No Bad Channels",
                        section=main_section_title, tags=("BadChannels", "NoBads"))
                
                # Update bads in temp_raw_for_psd for subsequent PSD plots
                temp_raw_for_psd.info['bads'] = bads_detected 
                self._add_psd_plot_to_report(
                    report, temp_raw_for_psd, "PSD: After Marking Bad Channels", "BadChannelsPSD",
                    main_section_title, status_update_prefix=f"Step {step_counter}: ")
            
            # At this point, an average reference is typically set. temp_raw_for_psd should reflect this.
            # The worker applies it before ICA. We'll assume it's applied for subsequent PSDs.
            # If not already average, apply it to temp_raw_for_psd if it makes sense for PSD.
            if not temp_raw_for_psd.info['projs']: # Simple check if an average ref proj might be missing
                 try:
                    temp_raw_for_psd.set_eeg_reference('average', projection=True)
                    self.status_update.emit(
                        f"Applied average reference to temporary raw for PSD plots (if not already present).")
                 except Exception as e:
                    self.status_update.emit(f"Could not apply average reference to temp raw for PSD: {e}")

            # --- Step: ICA (if applied) ---
            if self.settings.get('apply_ica') and self.ica_object: # ica_object from worker
                step_counter += 1
                report.add_html(
                    f"<h2>Step {step_counter}: Independent Component Analysis (ICA)</h2>",
                    title=f"Step{step_counter}_ICAHeader", section=main_section_title, tags=("ICA", "Header"))
                
                n_excluded = len(self.ica_object.exclude)
                report.add_html(
                    f"<p>ICA ({self.settings.get('ica_method', 'fastica')}) was applied. {n_excluded} components were marked for exclusion by ICLabel based on selected criteria.</p>",
                                title="ICA Status", section=main_section_title, tags=("ICA", "status"))

                if self.ica_object.exclude:
                    try:
                        # Plot component topographies of excluded components
                        fig_ica_topo = self.ica_object.plot_components(
                            picks=self.ica_object.exclude[:min(15, n_excluded)], show=False)
                        if isinstance(fig_ica_topo, list): # plot_components can return a list of figs
                             for i, fig in enumerate(fig_ica_topo):
                                report.add_figure(
                                    fig, title=f"ICA Excluded Component Topographies (Set {i+1})",
                                    section=main_section_title, tags=("ICA", "topomap_excluded"))
                                plt.close(fig)
                        else:
                            report.add_figure(
                                fig_ica_topo, title="ICA Excluded Component Topographies",
                                section=main_section_title, tags=("ICA", "topomap_excluded"))
                            plt.close(fig_ica_topo)
                    except Exception as e:
                        self.status_update.emit(f"Could not plot ICA component topographies for report: {e}")
                        report.add_html(
                            f"<p style=\'color:red;\'>Could not plot ICA component topographies: {e}</p>",
                            title="ICA Topography Error", section=main_section_title, tags=("ICA", "error"))
                    
                    # Plot sources of a few excluded components using original data (or a copy at that stage)
                    # For the report, we use raw_orig to show what these components looked like in the less processed data.
                    try:
                        raw_for_ica_sources = raw_orig.copy().load_data() # Use a less processed version for source viz
                        # If ICA was fit on filtered data, filter this copy similarly for plot_sources
                        if self.settings.get('apply_bandpass'):
                            raw_for_ica_sources.filter(
                                l_freq=self.settings.get('highpass_freq'), h_freq=self.settings.get('lowpass_freq')
                            )

                        num_excluded_to_plot = min(len(self.ica_object.exclude), 3)
                        if num_excluded_to_plot > 0:
                            fig_ica_sources = self.ica_object.plot_sources(
                                raw_for_ica_sources,
                                picks=self.ica_object.exclude[:num_excluded_to_plot],
                                show_scrollbars=False, show=False,
                                title=f"Sources of First {num_excluded_to_plot} Excluded ICA Components"
                            )
                            report.add_figure(
                                fig_ica_sources,
                                title=f"Sources of First {num_excluded_to_plot} Excluded ICA Components",
                                section=main_section_title, tags=("ICA", "sources_excluded")
                            )
                            plt.close(fig_ica_sources)
                    except Exception as e:
                        self.status_update.emit(f"Could not plot ICA sources for report: {e}")
                        report.add_html(
                            f"<p style=\'color:red;\'>Could not plot ICA sources: {e}</p>",
                            title="ICA Sources Error", section=main_section_title, tags=("ICA", "error"))
                else:
                    report.add_html(
                        "<p>No ICA components were excluded.</p>", title="ICA No Exclusions",
                        section=main_section_title, tags=("ICA", "info"))
                
                # Apply ICA to temp_raw_for_psd
                temp_raw_for_psd = self.ica_object.apply(temp_raw_for_psd.copy())
                self._add_psd_plot_to_report(
                    report, temp_raw_for_psd, "PSD: After ICA Application", "ICAPSD",
                    main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Step: ASR (if applied) ---
            if self.settings.get('apply_asr'):
                step_counter += 1
                asr_cutoff = self.settings.get('asr_cutoff', 'N/A')
                report.add_html(f"<h2>Step {step_counter}: Artifact Subspace Reconstruction (ASR)</h2>"
                                f"<p>ASR was applied with cutoff: {asr_cutoff}. "
                                f"ASR attempts to clean data segments with high-amplitude noise exceeding the calibration threshold. "
                                f"Its effects will be visible in subsequent PSD plots.</p>", 
                                title=f"Step{step_counter}_ASR", section=main_section_title, tags=("ASR", "status"))

            # --- Step: Channel Interpolation (if applied) ---
            # Note: raw_processed has interpolation if it happened. bads_detected are pre-interpolation.
            if self.settings.get('interpolate_bads') and bads_detected:
                step_counter += 1
                report.add_html(
                    f"<h2>Step {step_counter}: Bad Channel Interpolation</h2>"
                    f"<p>Bad channels ({', '.join(bads_detected)}) were interpolated using spherical splines.</p>",
                    title=f"Step{step_counter}_Interpolation", section=main_section_title,
                    tags=("Interpolation", "info"))
                # temp_raw_for_psd here might not have interpolation if it wasn't the last step.
                # For the PSD, we should use a version of data that HAS interpolation.
                # Let's use raw_processed for this PSD as it's the most complete version.
                # Or, we can try to interpolate temp_raw_for_psd if its bads are set.
                if temp_raw_for_psd.info['bads']:
                    try:
                        # temp_raw_for_psd.interpolate_bads(reset_bads=True, mode='accurate') # Simpler call
                        temp_raw_for_psd.interpolate_bads(reset_bads=True)
                        self._add_psd_plot_to_report(
                            report, temp_raw_for_psd, "PSD: After Channel Interpolation",
                            "InterpolationPSD", main_section_title, status_update_prefix=f"Step {step_counter}: ")
                    except Exception as e:
                        report.add_html(
                            f"<p style=\'color:red;\'>Error interpolating for report PSD: {e}</p>",
                            title="Interpolation PSD Error", section=main_section_title,
                            tags=("Interpolation", "error"))
                else:
                    self._add_psd_plot_to_report(
                        report, raw_processed, "PSD: After Channel Interpolation (from final data)",
                        "InterpolationPSD", main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Step: Final Processed Data ---
            step_counter += 1
            report.add_html(
                f"<h2>Step {step_counter}: Final Processed Data</h2>", title=f"Step{step_counter}_FinalDataHeader",
                section=main_section_title, tags=("FinalData", "Header"))
            report.add_raw(
                raw_processed, title="Final Processed Raw Data Snippet", psd=False, tags=("FinalData", "raw_snippet"))
            if raw_processed.get_montage():
                fig_sensors_proc = raw_processed.plot_sensors(show_names=True, show=False)
                report.add_figure(
                    fig_sensors_proc, title="Final Sensor Locations", section=main_section_title,
                    tags=("FinalData", "montage"))
                plt.close(fig_sensors_proc)
            self._add_psd_plot_to_report(
                report, raw_processed, "PSD: Final Processed Data", "FinalDataPSD",
                main_section_title, status_update_prefix=f"Step {step_counter}: ")

            # --- Overall PSD Comparison: Before vs. After ---
            step_counter += 1  # For logical flow, though it's a summary plot
            report.add_html(f"<h2>Step {step_counter}: Overall PSD Comparison: Before vs. After Processing</h2>",
                            title=f"Step{step_counter}_PSDBeforeAfterHeader",
                            section=main_section_title,
                            tags=("ComparisonPSD", "Header"))
            try:
                self.status_update.emit("Generating Before vs. After PSD comparison plot...")
                fig_comp, (ax_before, ax_after) = plt.subplots(1, 2, figsize=(14, 6))

                # Compute PSDs with error handling
                try:
                    # Pick only EEG channels
                    picks_before = mne.pick_types(raw_orig.info, eeg=True, exclude='bads')
                    picks_after = mne.pick_types(raw_processed.info, eeg=True, exclude='bads')

                    if len(picks_before) == 0 or len(picks_after) == 0:
                        raise ValueError("No EEG channels available for PSD comparison")

                    # Compute PSDs
                    psd_before = raw_orig.compute_psd(fmin=1, fmax=40, picks=picks_before)
                    psd_after = raw_processed.compute_psd(fmin=1, fmax=40, picks=picks_after)

                    # Get the data from PSD objects - each has its own frequency array
                    freqs_before = psd_before.freqs
                    freqs_after = psd_after.freqs
                    psd_before_data = psd_before.get_data().mean(axis=0)  # Average across channels
                    psd_after_data = psd_after.get_data().mean(axis=0)

                    # Convert to dB, handling potential zeros/negative values
                    with np.errstate(divide='ignore', invalid='ignore'):
                        psd_before_db = 10 * np.log10(psd_before_data + 1e-12)  # Add small value to avoid log(0)
                        psd_after_db = 10 * np.log10(psd_after_data + 1e-12)

                    # Plot Before Processing PSD with its own frequencies
                    ax_before.plot(freqs_before, psd_before_db, color='blue', alpha=0.8, linewidth=2)
                    ax_before.set_title(f"Before Processing (fs={raw_orig.info['sfreq']} Hz)", fontsize=12,
                                        fontweight='bold')
                    ax_before.set_xlabel("Frequency (Hz)")
                    ax_before.set_ylabel("Power Spectral Density (dB/Hz)")
                    ax_before.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                    ax_before.set_xlim(1, 40)

                    # Plot After Processing PSD with its own frequencies
                    ax_after.plot(freqs_after, psd_after_db, color='red', alpha=0.8, linewidth=2)
                    ax_after.set_title(f"After Processing (fs={raw_processed.info['sfreq']} Hz)", fontsize=12,
                                       fontweight='bold')
                    ax_after.set_xlabel("Frequency (Hz)")
                    ax_after.set_ylabel("Power Spectral Density (dB/Hz)")
                    ax_after.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)
                    ax_after.set_xlim(1, 40)

                    # Make y-axis limits the same for both plots for easier comparison
                    y_min = min(psd_before_db.min(), psd_after_db.min()) - 2
                    y_max = max(psd_before_db.max(), psd_after_db.max()) + 2
                    ax_before.set_ylim(y_min, y_max)
                    ax_after.set_ylim(y_min, y_max)

                    # Add overall title
                    fig_comp.suptitle("Average EEG Power Spectral Density Comparison", fontsize=14, fontweight='bold')

                    # Adjust layout
                    fig_comp.tight_layout()

                    report.add_figure(fig_comp, title="PSD Comparison: Before vs. After", section=main_section_title,
                                      tags=("ComparisonPSD", "Plot"))
                    plt.close(fig_comp)
                    self.status_update.emit("Before vs. After PSD comparison plot added.")

                except Exception as e:
                    plt.close(fig_comp)  # Clean up the figure
                    raise e

            except Exception as e:
                error_msg = f"Could not generate Before vs. After PSD comparison plot: {e}"
                self.status_update.emit(error_msg)
                report.add_html(f"<p style=\'color:red;\'>{error_msg}</p>",
                                title="Comparison PSD Error",
                                section=main_section_title, tags=("ComparisonPSD", "error"))

            # --- Save report ---
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
            report_filename = target_report_dir / f"{Path(original_filename).stem}_preprocessing_report.html"
            
            self.status_update.emit(f"Attempting to save report to: {str(report_filename)}")
            try:
                report.save(str(report_filename), overwrite=True, open_browser=False)
                self.status_update.emit(f"Report successfully saved to {report_filename}")
                report_generation_successful = True
            except Exception as e_save:
                save_err_msg = f"CRITICAL: Failed to save report to {report_filename}. Error: {e_save}"
                self.status_update.emit(save_err_msg)
                self.processing_error.emit(original_filename, save_err_msg)

        except Exception as e_generate:
            # Catch any other exception during report content generation
            gen_err_msg = f"Error during report content generation for {original_filename}: {e_generate}"
            self.status_update.emit(gen_err_msg)
            self.processing_error.emit(original_filename, gen_err_msg)

        # return report_generation_successful

    def _add_psd_plot_to_report(self, report: mne.Report, raw_object: mne.io.Raw, 
                                title_prefix: str, tag: str, main_section_title: str, 
                                status_update_prefix: str = ""):
        """Helper function to compute and add a PSD plot to the report."""
        try:
            self.status_update.emit(f"{status_update_prefix}Generating PSD: {title_prefix}...")
            # Ensure data is loaded for PSD, operate on a copy to be safe
            psd_plot_raw = raw_object.copy().load_data() 
            
            # Pick only EEG channels, excluding any marked as bad FOR THIS PSD PLOT
            # This ensures PSD is computed on what's considered "good" at this stage for plotting purposes
            # If all channels are bad or no EEG channels, it will be handled.
            eeg_picks = mne.pick_types(psd_plot_raw.info, eeg=True, exclude=psd_plot_raw.info['bads'])
            
            if len(eeg_picks) == 0:
                self.status_update.emit(f"Skipping PSD plot for '{title_prefix}': No good EEG channels found.")
                report.add_html(f"<p><i>Skipping PSD plot for '{title_prefix}': No good EEG channels found.</i></p>", 
                                title=f"Skipped PSD Plot Info: {tag}",
                                section=main_section_title, tags=(tag, "info"))
                return

            fig_psd = psd_plot_raw.compute_psd(
                fmin=1, fmax=80, n_fft=min(2048, int(psd_plot_raw.info['sfreq'])),
                n_overlap=int(min(2048, int(psd_plot_raw.info['sfreq'])) / 2),
                picks=eeg_picks).plot(show=False)
            report.add_figure(fig_psd, title=title_prefix, section=main_section_title, tags=(tag, "psd"))
            plt.close(fig_psd)
            self.status_update.emit(f"PSD for '{title_prefix}' added to report.")
        except Exception as e:
            error_msg = f"Could not generate PSD plot for '{title_prefix}': {e}"
            self.status_update.emit(error_msg)
            report.add_html(f"<p style=\'color:red;\'>{error_msg}</p>", 
                            title=f"PSD Plot Error: {tag}",
                            section=main_section_title, tags=(tag, "error"))


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
        self.log_lineedit.setText(message) # Set plain text, HTML tags will not be rendered.

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
        """Find EEG files in directory"""
        files = []
        path = Path(directory)

        for ext in extensions:
            if pattern:
                files.extend(path.rglob(f"*{pattern}*{ext}"))
            else:
                files.extend(path.rglob(f"*{ext}"))

        return [str(f) for f in files]

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
            self._create_montage_plot() # This will clear the plot area
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
            # Validate highpass filter frequency
            if self.step2_bp_filter_checkbox.isChecked():
                hp_freq = float(self.step2_hp_filter_lineedit.text())
                lp_freq = float(self.step2_lp_filter_lineedit.text())
                if hp_freq <= 0 or lp_freq <= 0:
                    raise ValueError("Filter frequency must be positive")

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
    controller.setWindowTitle("CleanEEG GUI")
    controller.showMaximized()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
