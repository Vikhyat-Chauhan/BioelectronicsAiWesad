#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Global variables & configuration
DATA_PATH = 'data/WESAD/'
chest_columns = ['sid', 'acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']
all_columns = [
    'sid', 'c_acc_x', 'c_acc_y', 'c_acc_z', 'ecg', 'emg', 'c_eda', 'c_temp', 'resp',
    'w_acc_x', 'w_acc_y', 'w_acc_z', 'bvp', 'w_eda', 'w_temp', 'label'
]
# Subject IDs used in the WESAD dataset
ids = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]

# Sampling frequencies
sf_BVP = 64
sf_EDA = 4
sf_TEMP = 4
sf_ACC = 32
sf_chest = 700

def map_chest_labels(label_array, output_length, chest_sf, target_sf):
    """
    Resample label_array from chest_sf to target_sf by taking the mode
    of the corresponding chest-label frames.
    """
    batch_size = chest_sf / target_sf
    mapped_labels = np.zeros((output_length, 1))
    
    for i in range(output_length):
        start_idx = round(i * batch_size)
        end_idx = round((i + 1) * batch_size) - 1
        mapped_labels[i] = stats.mode(label_array[start_idx:end_idx])[0].squeeze()
    
    return mapped_labels

def pkl_to_np_wrist(filename, subject_id):
    """
    Convert data from the WESAD pickle file into three NumPy arrays
    for wrist accelerometer, BVP, and EDA/TEMP. Resample the label 
    to match each signal's frequency.
    """
    unpickled = pd.read_pickle(filename)
    
    wrist_acc = unpickled["signal"]["wrist"]["ACC"]    # shape: (N_acc, 3)
    wrist_bvp = unpickled["signal"]["wrist"]["BVP"]    # shape: (N_bvp, 1)
    wrist_eda = unpickled["signal"]["wrist"]["EDA"]    # shape: (N_eda, 1)
    wrist_temp = unpickled["signal"]["wrist"]["TEMP"]  # shape: (N_eda, 1)
    
    labels_chest = unpickled["label"].reshape(-1, 1)   # original chest labels

    # 1) Wrist Accelerometer data
    sid_acc = np.repeat(subject_id, len(wrist_acc)).reshape(-1, 1)
    lbl_acc = map_chest_labels(labels_chest, len(wrist_acc), sf_chest, sf_ACC)
    data_acc = np.concatenate([sid_acc, wrist_acc, lbl_acc], axis=1)

    # 2) Wrist BVP data
    sid_bvp = np.repeat(subject_id, len(wrist_bvp)).reshape(-1, 1)
    lbl_bvp = map_chest_labels(labels_chest, len(wrist_bvp), sf_chest, sf_BVP)
    data_bvp = np.concatenate([sid_bvp, wrist_bvp, lbl_bvp], axis=1)

    # 3) Wrist EDA & TEMP data
    sid_eda_temp = np.repeat(subject_id, len(wrist_eda)).reshape(-1, 1)
    lbl_eda_temp = map_chest_labels(labels_chest, len(wrist_eda), sf_chest, sf_EDA)
    data_eda_temp = np.concatenate([sid_eda_temp, wrist_eda, wrist_temp, lbl_eda_temp], axis=1)

    return data_acc, data_bvp, data_eda_temp

def merge_wrist_data():
    """
    Read the data for each subject, convert the wrist signals to
    NumPy arrays, and concatenate them into three final arrays.
    Then save them as pickle files.
    """
    merged_acc = []
    merged_bvp = []
    merged_eda_temp = []
    
    for sid in ids:
        file = f"{DATA_PATH}S{sid}/S{sid}.pkl"
        data_acc, data_bvp, data_eda_temp = pkl_to_np_wrist(file, sid)
        
        merged_acc.append(data_acc)
        merged_bvp.append(data_bvp)
        merged_eda_temp.append(data_eda_temp)

    # Concatenate all subjects
    merged_acc = np.concatenate(merged_acc, axis=0)
    merged_bvp = np.concatenate(merged_bvp, axis=0)
    merged_eda_temp = np.concatenate(merged_eda_temp, axis=0)

    # Save to pickle
    cols_acc = ['sid', 'w_acc_x', 'w_acc_y', 'w_acc_z', 'label']
    cols_bvp = ['sid', 'bvp', 'label']
    cols_eda_temp = ['sid', 'w_eda', 'w_temp', 'label']

    pd.DataFrame(merged_acc, columns=cols_acc).to_pickle('data/subj_merged_acc_w.pkl')
    pd.DataFrame(merged_bvp, columns=cols_bvp).to_pickle('data/subj_merged_bvp_w.pkl')
    pd.DataFrame(merged_eda_temp, columns=cols_eda_temp).to_pickle('data/subj_merged_eda_temp_w.pkl')

def pkl_to_np_chest(filename, subject_id):
    """
    Convert the chest data from a WESAD pickle file to a NumPy array,
    attaching subject_id as the first column and label as the last column.
    """
    unpickled = pd.read_pickle(filename)
    
    chest_acc = unpickled["signal"]["chest"]["ACC"]   # shape: (N, 3)
    chest_ecg = unpickled["signal"]["chest"]["ECG"]   # shape: (N, 1)
    chest_emg = unpickled["signal"]["chest"]["EMG"]   # shape: (N, 1)
    chest_eda = unpickled["signal"]["chest"]["EDA"]   # shape: (N, 1)
    chest_temp = unpickled["signal"]["chest"]["Temp"] # shape: (N, 1)
    chest_resp = unpickled["signal"]["chest"]["Resp"] # shape: (N, 1)
    lbl = unpickled["label"].reshape(-1, 1)
    
    sid_array = np.full((len(lbl), 1), subject_id)
    
    # Concatenate into a single array: 
    # [sid, acc1, acc2, acc3, ecg, emg, eda, temp, resp, label]
    chest_all = np.concatenate([
        sid_array,
        chest_acc, chest_ecg, chest_emg,
        chest_eda, chest_temp, chest_resp,
        lbl
    ], axis=1)

    return chest_all

def merge_chest_data():
    """
    Read the data for each subject, convert the chest signals to
    NumPy arrays, and concatenate them into one final array.
    Then save the result as a pickle file.
    """
    merged = []
    for sid in ids:
        file = f"{DATA_PATH}S{sid}/S{sid}.pkl"
        chest_data = pkl_to_np_chest(file, sid)
        merged.append(chest_data)
    
    merged = np.concatenate(merged, axis=0)
    pd.DataFrame(merged, columns=chest_columns).to_pickle('data/merged_chest.pkl')

def filter_chest_data():
    """
    Load the merged chest data, keep only rows where label is in {1,2,3}
    and temperature is greater than 0, then save as a new pickle file.
    """
    df = pd.read_pickle("data/merged_chest.pkl")
    df_filtered = df[df["label"].isin([1, 2, 3])]
    df_filtered = df_filtered[df_filtered["temp"] > 0]
    df_filtered.to_pickle("data/merged_chest_fltr.pkl")

def preprocess():
    """
    High-level function to execute all preprocessing steps.
    """
    merge_wrist_data()
    merge_chest_data()
    filter_chest_data()
    print("Preprocessing completed.")

# Run the full preprocessing pipeline
if __name__ == "__main__":
    preprocess()
