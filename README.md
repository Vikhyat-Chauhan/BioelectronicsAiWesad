# Stress Detection from Wearable Biosensors using Deep Autoencoders

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM-green.svg)](https://scikit-learn.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-red.svg)](https://www.gnu.org/licenses/gpl-3.0)

A machine learning pipeline for **three-class emotion classification** (neutral, stress, amusement) from multimodal physiological signals collected by wearable biosensors. Uses convolutional autoencoders for unsupervised latent feature extraction and SVM for classification, achieving **83.5% accuracy** on the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection).

---

## Key Results

| Sensor Configuration | Accuracy | F1-Score |
|---|---|---|
| **Chest + Wrist (Combined)** | **83.5% (+/- 11.1%)** | **81.1% (+/- 12.5%)** |
| Chest Only | 80.7% (+/- 10.7%) | 78.0% (+/- 10.3%) |
| Wrist Only | 75.0% (+/- 12.9%) | 75.0% (+/- 12.9%) |

Evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation across 15 subjects.

---

## Architecture

```
                          WESAD Raw Signals (15 subjects)
                                     |
                    +----------------+----------------+
                    |                |                |
              Chest Sensors    Wrist BVP       Wrist EDA/TEMP
            (ECG,EMG,EDA,      (64 Hz)          (4 Hz)
             TEMP,RESP)
              700 Hz
                    |                |                |
              Preprocessing: resampling, label mapping, filtering
                    |                |                |
              +-----+-----+   +-----+-----+   +-----+-----+
              | Conv1D    |   | Conv1D    |   | Conv1D    |
              | Autoencoder|  | Autoencoder|  | Autoencoder|
              | (80-dim)  |   | (40-dim)  |   | (4-dim)   |
              +-----------+   +-----------+   +-----------+
                    |                |                |
                    +-------+--------+--------+-------+
                            |                 |
                     124-dim latent feature vector
                            |
                    +-------+-------+
                    |  SVM Classifier |
                    |  (poly kernel) |
                    +-------+-------+
                            |
                  Neutral / Stress / Amusement
```

---

## Project Structure

```
.
├── preprocessing.py           # Data loading, resampling, and merging pipeline
├── latentfeatureextractor.py  # Convolutional autoencoder models and feature extraction
├── svmclassifier.ipynb        # SVM classification, evaluation, and visualization
├── requirements.txt           # Python dependencies
├── LICENSE                    # GNU GPL v3
└── README.md
```

---

## How It Works

### 1. Preprocessing (`preprocessing.py`)
- Loads raw WESAD pickle files for 15 subjects (wrist + chest sensors)
- Resamples labels from chest frequency (700 Hz) to each sensor's native frequency using statistical mode
- Merges all subjects into unified DataFrames and filters invalid samples

### 2. Feature Extraction (`latentfeatureextractor.py`)
- Builds three **1D convolutional autoencoders** (one per sensor modality) with batch normalization
- Segments signals into 0.25-second windows
- Trains using LOSO cross-validation -- each fold holds out one subject for testing
- Extracts compressed latent representations:
  - Chest encoder: 5 signals -> **80 features**
  - BVP encoder: 1 signal -> **40 features**
  - EDA/TEMP encoder: 2 signals -> **4 features**
- Concatenates into a **124-dimensional feature vector** per window

### 3. Classification (`svmclassifier.ipynb`)
- Trains an SVM classifier (polynomial kernel, degree 3) on the extracted features
- Evaluates three configurations: combined, chest-only, and wrist-only
- Generates per-subject confusion matrices and comparative bar charts

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Dataset

Download the [WESAD dataset](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection) and place it in `data/WESAD/` so the structure is:

```
data/WESAD/
├── S2/S2.pkl
├── S3/S3.pkl
├── ...
└── S17/S17.pkl
```

### Run the Pipeline

```bash
# Step 1: Preprocess and merge subject data
python preprocessing.py

# Step 2: Train autoencoders and extract latent features
python latentfeatureextractor.py

# Step 3: Open the notebook for classification and results
jupyter notebook svmclassifier.ipynb
```

---

## Tech Stack

- **Deep Learning**: TensorFlow / Keras (Conv1D autoencoders)
- **Machine Learning**: scikit-learn (SVM, StandardScaler, metrics)
- **Data Processing**: NumPy, pandas, SciPy
- **Visualization**: Matplotlib, Seaborn

---

## Dataset Reference

> Schmidt, P., Reiss, A., Duerichen, R., Marber, C., & Van Laerhoven, K. (2018).
> *Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection.*
> ICMI 2018.

---

## License

This project is licensed under the GNU General Public License v3.0 -- see [LICENSE](LICENSE) for details.
