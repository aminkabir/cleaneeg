# CleanEEG - Automated Python-based Resting-State EEG Preprocessing GUI
**CleanEEG** is a comprehensive application that democratizes professional-grade EEG preprocessing for researchers and clinicians. Built on the proven MNE-Python ecosystem and industry-standard algorithms, it transforms complex signal processing workflows into an intuitive point-and-click interface.

**Perfect for:** Researchers transitioning from manual preprocessing, labs seeking standardized workflows, and anyone needing publication-ready EEG data without the programming overhead. CleanEEG bridges the gap between research-grade signal processing and practical usability, ensuring your EEG analysis starts with the cleanest possible data.

Built on the resting-state EEG preprocessing pipeline proposed by DISCOVER-EEG[^1]—a peer-reviewed, standardized framework originally implemented in MATLAB—this Python implementation ensures reproducible, publication-ready results that align with established best practices in EEG research.

## Installation

Below are two ways to set up a new environment and install all required dependencies: using pip (with a virtual environment) or conda.

-  Clone or download the CleanEEG repository and navigate into it:

```
git clone https://github.com/aminkabir/cleaneeg
cd cleaneeg
```

### Using pip (Python’s built-in venv)

- Create and activate a virtual environment:

```
# Create a new virtual environment named "cleaneeg"
python3 -m venv cleaneeg

# Activate the environment
# On Linux/macOS:
source cleaneeg/bin/activate
# On Windows (PowerShell):
.\cleaneeg\Scripts\Activate.ps1
# On Windows (cmd.exe):
# .\cleaneeg\Scripts\activate.bat
```

- Install all required packages from requirements.txt:

```
pip install --upgrade pip
pip install -r requirements.txt
```

### Using conda

- Create and activate a new conda environment from environment.yml:

```
conda env create -f environment.yml
conda activate cleaneeg
```

### Launch the GUI:

- Navigate into the CleanEEG directory (if not already there):

```
cd cleaneeg
```

- Launch the GUI:

```
python cleaneeg_gui.py
```

## Step-by-Step Usage

### 1. Load EEG Data
- Click "Select EEG Data Directory for Processing"
- Choose data format from dropdown (or use "Automatically find EEG files")
- Find data:
  - "Find All Available EEG Data" - loads all compatible files
  - "Find EEG Data with Specific Naming Convention" - filter by filename pattern
- Click "Import Data"[^2]

### 2. Configure Montage
- Select standard montage from dropdown (e.g., standard_1020)
- Or choose "Use Custom Montage" and load your own channel location file
- Preview appears in the right panel when you select a file

### 3. Select Preprocessing Steps
Check/uncheck the desired preprocessing options:

- **Line Noise Removal (DSS)[^3] - Remove 50/60 Hz interference**

- **Highpass Filter - Default: 1 Hz[^4]**

- **Downsample Data - Default: 500 Hz[^5]**

- **Bad Channel Rejection (PREP Pipeline)[^6]**

- **Independent Component Analysis (ICA)[^7]**

- **Auto-classify components (ICLabel)[^8]: Muscle, Eye Blink, Heart Beat, Others**

- **Bad Channel Interpolation[^9]**

- **Bad Time Segments Removal (ASR)[^10]**

### 4. Configure Output
- Select output format (Auto, BrainVision, EEGLAB, or EDF)[^11]
- Click "Select Save Location" to choose output directory
- Check "Export Report" for detailed HTML preprocessing reports

### 5. Run Processing
- Click "Auto-Clean All Resting-State EEG"
- Monitor progress in the status bar
- Check log messages for real-time updates

## cleaneeg_tutorial.ipynb

For users new to EEG preprocessing or CleanEEG, open cleaneeg_tutorial.ipynb in Jupyter Notebook or JupyterLab. This tutorial provides:

- Background on preprocessing steps.

- Code examples illustrating each step (loading data, filtering, ICA, etc.).

- Best practices for artifact detection and removal.

- Exporting and inspecting cleaned data.

Running through the notebook ensures you understand how CleanEEG processes resting-state EEG and how to adjust parameters for your specific dataset.

### References:

[^1]: https://github.com/crisglav/discover-eeg

[^2]: https://mne.tools/stable/generated/mne.io.read_raw.html

[^3]: https://nbara.github.io/python-meegkit/modules/meegkit.dss.html

[^4]: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.filter

[^5]: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.resample

[^6]: https://pyprep.readthedocs.io/en/latest/generated/pyprep.NoisyChannels.html

[^7]: https://mne.tools/stable/generated/mne.preprocessing.ICA.html

[^8]: https://mne.tools/mne-icalabel/stable/index.html

[^9]: https://mne.tools/stable/generated/mne.io.Raw.html#mne.io.Raw.interpolate_bads

[^10]: https://nbara.github.io/python-meegkit/modules/meegkit.asr.html

[^11]: https://mne.tools/stable/generated/mne.export.export_raw.html
