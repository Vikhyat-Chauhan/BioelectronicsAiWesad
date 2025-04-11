# BioelectronicsAiWesad
 Bioelectronics project, implementing machine learning for detecting stress using data obtained by biosensors(WESAD dataset).

## Preprocess and merge subject data
### python merge_subj_data.py

## Input data path: 'data/WESAD/'
## Generates the following files in data folder:
### subj_merged_acc_w.pkl
### subj_merged_bvp_w.pkl
### subj_merged_eda_temp_w.pkl
### merged_chest_fltr.pkl

## Create autoencoder model and extract latent features
### python extract_ae_latent_features.py

## Input files:
### subj_merged_acc_w.pkl
### subj_merged_bvp_w.pkl
### subj_merged_eda_temp_w.pkl
### merged_chest_fltr.pkl

## ae_feature_extractor.py 
### To build and train autoencoder model and extract features.
### Save extracted features leaving one subject out into pickle files in features/train and features/test directories. The number in the filename indicates which subject was left out in each fold.

## SVM_classifier.ipynb
### Build SVM classifier that uses latent features extracted by autoencoder for three class classification of WESAD dataset: neutral, stress, and ammusement. Results analysis also included.

## MLP_classifier.ipynb
### Build MLP (Multi Layer Perceptron) classifier that uses latent features extracted by autoencoder for three class classification of WESAD dataset: neutral, stress, and ammusement. Results analysis also included.