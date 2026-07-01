# Datasets Used in the Project

This document tracks the public datasets utilized in the Industrial Predictive Maintenance Platform.

## 1. NASA C-MAPSS (Turbofan Engine Degradation Simulation)
- **Source**: NASA Prognostics Data Repository (or academic mirrors).
- **License**: Public Domain (US Government Work).
- **Structure**: Text files containing simulated engine degradation sensor readings. Divided into 4 sub-datasets (FD001, FD002, FD003, FD004) with different operating conditions and fault modes.
- **Project Use Case**: Primary tabular dataset for Phase 1. We use it to predict the Remaining Useful Life (RUL) of an engine using baseline ML models and advanced sequence models.
- **Citation**: A. Saxena, K. Goebel, D. Simon, and N. Eklund, "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", in the Proceedings of the 1st International Conference on Prognostics and Health Management (PHM08), Denver CO, Oct 2008.

## 2. MIMII (Malfunctioning Industrial Machine Investigation and Inspection)
- **Source**: [Zenodo (ID: 3384388)](https://zenodo.org/records/3384388)
- **License**: Creative Commons Attribution 4.0 International
- **Structure**: WAV audio files. The dataset includes normal and anomalous sounds for 4 machine types (valves, pumps, fans, slide rails).
- **Project Use Case**: Secondary modality for Phase 4. Used to build an audio-based anomaly detection model (autoencoder) to flag machine faults from sound.
- **Citation**: Harsh Purohit, Ryo Tanabe, Kenji Ichige, Takashi Endo, Yuki Nikaido, Kaori Suefusa, and Yohei Kawaguchi, "MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection," in Proc. 4th Workshop on Detection and Classification of Acoustic Scenes and Events (DCASE), 2019.

## 3. CWRU Bearing Data
- **Source**: Case Western Reserve University Bearing Data Center
- **License**: Free for research use.
- **Structure**: MATLAB (`.mat`) files containing vibration signal recordings from a motor under various fault conditions (inner race, outer race, ball defects).
- **Project Use Case**: Additional modality for future phases, or supplementing baseline testing with a real-world high-frequency vibration dataset.
- **Citation**: Smith, W. A., & Randall, R. B. (2015). Rolling element bearing diagnostics using the Case Western Reserve University data: A benchmark study. Mechanical Systems and Signal Processing, 64, 100-131.
