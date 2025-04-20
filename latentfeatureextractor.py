#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, Dense, Flatten, Reshape, BatchNormalization, Conv1D
from keras.models import Model
from keras.layers import ReLU
from keras.optimizers import RMSprop
import tensorflow as tf

# -------------------------------------------------------------------
# File paths and constants
# -------------------------------------------------------------------
C_PATH = "data/merged_chest_fltr.pkl"
W1_PATH = "data/subj_merged_bvp_w.pkl"
W2_PATH = "data/subj_merged_eda_temp_w.pkl"

CHEST_FEATURES = ['ecg', 'emg', 'eda', 'temp', 'resp']  # sampling frequency: 700
BVP_FEATURES = ['bvp']                                  # sampling frequency: 64
EDA_TEMP_FEATURES = ['w_eda', 'w_temp']                 # sampling frequency: 4

SF_CHEST = 700
SF_BVP = 64
SF_EDA = 4
SF_TEMP = 4

WINDOW_SECONDS = 0.25  # sampling window (in seconds)

# -------------------------------------------------------------------
# Class Definition
# -------------------------------------------------------------------
class PhysioAutoencoder:
    """
    This class encapsulates the process of:
      1. Loading and preparing physiological datasets (chest, BVP, EDA/TEMP).
      2. Training autoencoder models for chest data using leave-one-subject-out.
      3. Extracting encoded features (embeddings) for all modalities.
    """

    def __init__(self):
        """Initialize and pre-process the dataframes."""
        # Load data
        self.df_c = pd.read_pickle(C_PATH)
        self.df_w1 = pd.read_pickle(W1_PATH)  # BVP
        self.df_w2 = pd.read_pickle(W2_PATH)  # EDA + TEMP

        # Keep only relevant classes
        self.df_w1 = self.df_w1[self.df_w1["label"].isin([1, 2, 3])]
        self.df_w2 = self.df_w2[self.df_w2["label"].isin([1, 2, 3])]

        # Compute batch sizes for each device (number of samples in each window)
        self.batch_size_chest = int(SF_CHEST * WINDOW_SECONDS)
        self.batch_size_bvp = int(SF_BVP * WINDOW_SECONDS)
        self.batch_size_eda = int(SF_EDA * WINDOW_SECONDS)
        self.batch_size_temp = int(SF_TEMP * WINDOW_SECONDS)  # same as EDA here

        # Unique subject IDs
        self.subject_ids = self.df_c["sid"].unique().astype(int)

        # Number of classes (e.g., 3 stress levels)
        self.num_classes = len(self.df_c["label"].unique())

    def encode_one_hot(self, labels: np.ndarray, num_classes: int) -> np.ndarray:
        """
        Convert numeric labels into a one-hot-encoded matrix.
        
        :param labels: Array of label integers.
        :param num_classes: The total number of classes.
        :return: One-hot-encoded array with shape (num_samples, num_classes).
        """
        encoded = np.zeros((labels.shape[0], num_classes))
        for i, val in enumerate(labels):
            encoded[i, val - 1] = 1
        return encoded

    def get_data(
        self,
        test_id: int,
        batch_size: int,
        feature_list: list,
        df: pd.DataFrame
    ):
        """
        Prepare training and testing data for a specific modality using a leave-one-subject-out approach.
        
        :param test_id: The subject ID to be used as test set.
        :param batch_size: Number of samples to form a single window.
        :param feature_list: List of feature column names.
        :param df: The dataframe containing the raw signals and labels.
        :return: 
            X_train: Training feature windows
            Y_train: One-hot training labels
            X_test: Testing feature windows
            Y_test: One-hot testing labels
            train_labels_int: Integer training labels for each window
            test_labels_int: Integer testing labels for each window
        """
        merged_train, merged_train_y, merged_train_labels = None, None, None
        X_test, Y_test, test_labels_int = None, None, None
        merge_init_flag = False

        # Loop over each subject
        for subject_id in self.subject_ids:
            df_subject = df[df["sid"] == subject_id]

            # Truncate so that length is a multiple of batch_size
            n = (len(df_subject) // batch_size) * batch_size
            df_subject = df_subject[:n]

            # Scale features
            scaled_subject = StandardScaler().fit_transform(df_subject[feature_list])

            # Reshape into windows of shape (num_windows, n_features, batch_size)
            reshaped = scaled_subject.reshape(
                int(scaled_subject.shape[0] / batch_size),
                scaled_subject.shape[1],
                batch_size
            )

            # Determine the label for each window by mode of included samples
            label_values = df_subject["label"].values.astype(int)
            window_labels = np.zeros((reshaped.shape[0], 1))

            for i in range(reshaped.shape[0]):
                window_labels[i] = stats.mode(
                    label_values[i * batch_size : (i + 1) * batch_size]
                )[0].squeeze()

            window_labels = window_labels.astype(int)
            one_hot_labels = self.encode_one_hot(window_labels, self.num_classes).astype(int)

            # Assign to train or test
            if subject_id == test_id:
                X_test = reshaped
                Y_test = one_hot_labels
                test_labels_int = window_labels
            else:
                # Merge with existing training data
                if not merge_init_flag:
                    merged_train = reshaped
                    merged_train_y = one_hot_labels
                    merged_train_labels = window_labels
                    merge_init_flag = True
                else:
                    merged_train = np.concatenate((merged_train, reshaped), axis=0)
                    merged_train_y = np.concatenate((merged_train_y, one_hot_labels), axis=0)
                    merged_train_labels = np.concatenate((merged_train_labels, window_labels), axis=0)

        print("Train Data:", merged_train.shape, merged_train_y.shape)
        print(" Test Data:", X_test.shape, Y_test.shape)

        return merged_train, merged_train_y, X_test, Y_test, merged_train_labels, test_labels_int

    # -----------------------------------------------------------------------
    # Autoencoder Model Builders
    # -----------------------------------------------------------------------
    def build_autoencoder_chest(self, batch_size: int, num_features: int):
        """
        Construct an autoencoder model specialized for the chest device signals.
        
        :param batch_size: Window size (number of samples per window).
        :param num_features: Number of feature channels (ecg, emg, etc.).
        :return: (encoder, autoencoder_model)
        """
        input_signal = Input(shape=(num_features, batch_size))

        # Encoder
        x = Conv1D(batch_size, kernel_size=6, activation='relu', padding='same')(input_signal)
        x = BatchNormalization()(x)
        x = Conv1D(batch_size, kernel_size=3, activation='relu', padding='same')(x)
        x = Flatten()(x)
        encoded = Dense(80, activation='relu')(x)
        encoder = Model(input_signal, encoded)

        # Decoder
        d = Dense(batch_size * num_features)(encoded)
        d = Reshape((num_features, batch_size))(d)
        d = Conv1D(batch_size, kernel_size=3, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        decoded = Conv1D(batch_size, kernel_size=6, activation='sigmoid', padding='same')(d)

        autoencoder_model = Model(input_signal, decoded)
        return encoder, autoencoder_model

    def build_autoencoder_bvp(self, batch_size: int, num_features: int):
        """
        Construct an autoencoder model specialized for BVP signals.
        
        :param batch_size: Window size (number of samples per window).
        :param num_features: Number of feature channels (only BVP in this case).
        :return: (encoder, autoencoder_model)
        """
        input_signal = Input(shape=(num_features, batch_size))

        # Encoder
        x = Conv1D(batch_size, kernel_size=6, activation='relu', padding='same')(input_signal)
        x = BatchNormalization()(x)
        x = Conv1D(batch_size, kernel_size=3, activation='relu', padding='same')(x)
        x = Flatten()(x)
        encoded = Dense(40, activation='relu')(x)
        encoder = Model(input_signal, encoded)

        # Decoder
        d = Dense(batch_size * num_features)(encoded)
        d = Reshape((num_features, batch_size))(d)
        d = Conv1D(batch_size, kernel_size=3, activation='relu', padding='same')(d)
        d = BatchNormalization()(d)
        decoded = Conv1D(batch_size, kernel_size=6, activation='sigmoid', padding='same')(d)

        autoencoder_model = Model(input_signal, decoded)
        return encoder, autoencoder_model

    def build_autoencoder_eda_temp(self, batch_size: int, num_features: int):
        """
        Construct an autoencoder model specialized for EDA + TEMP signals.
        
        :param batch_size: Window size (number of samples per window).
        :param num_features: Number of feature channels (EDA and TEMP in this case).
        :return: (encoder, autoencoder_model)
        """
        input_signal = Input(shape=(num_features, batch_size))

        # Encoder
        x = Conv1D(batch_size, kernel_size=4, activation='relu', padding='same')(input_signal)
        x = BatchNormalization()(x)
        x = Flatten()(x)
        encoded = Dense(4, activation='relu')(x)
        encoder = Model(input_signal, encoded)

        # Decoder
        d = Dense(batch_size * num_features)(encoded)
        d = Reshape((num_features, batch_size))(d)
        decoded = Conv1D(batch_size, kernel_size=4, activation='sigmoid', padding='same')(d)

        autoencoder_model = Model(input_signal, decoded)
        return encoder, autoencoder_model

    # -----------------------------------------------------------------------
    # Training Autoencoders & Extracting Features
    # -----------------------------------------------------------------------
    def train_chest_autoencoders(self):
        """
        Train a chest autoencoder for each subject using a leave-one-subject-out approach.
        The model is saved per subject with a filename indicating the held-out subject.
        """
        for sid in self.subject_ids:
            print(f"\nTraining chest autoencoder (LOSO) for subject {sid} ...")
            X_train, Y_train, X_test, Y_test, _, _ = self.get_data(
                test_id=sid,
                batch_size=self.batch_size_chest,
                feature_list=CHEST_FEATURES,
                df=self.df_c
            )

            encoder, model = self.build_autoencoder_chest(
                batch_size=self.batch_size_chest,
                num_features=len(CHEST_FEATURES)
            )
            model.compile(optimizer=RMSprop(learning_rate=0.00025), loss="mse")

            # Train
            model.fit(X_train, X_train, epochs=10, verbose=1)

            # Save the encoder
            model_name = f"trained_models/c/encoder_loso_{sid}.h5"
            encoder.save(model_name)
            print(f"Encoder model saved: {model_name}")

    def extract_features(self):
        """
        Using the trained chest autoencoders, extract features (embeddings) from:
          1. Chest signals
          2. BVP signals
          3. EDA + TEMP signals
        Then, concatenate these embeddings with class labels and save the results.
        """
        for sid in self.subject_ids:
            print(f"\n========== Extracting features for subject {sid} ==========")

            # --------------------------
            # Chest
            # --------------------------
            X_train_chest, _, X_test_chest, _, train_labels_int, test_labels_int = self.get_data(
                test_id=sid,
                batch_size=self.batch_size_chest,
                feature_list=CHEST_FEATURES,
                df=self.df_c
            )

            chest_encoder_path = f"trained_models/c/encoder_loso_{sid}.h5"
            print(f"Loading chest encoder: {chest_encoder_path}")
            chest_encoder = tf.keras.models.load_model(chest_encoder_path)

            # --------------------------
            # BVP
            # --------------------------
            X_train_bvp, _, X_test_bvp, _, _, _ = self.get_data(
                test_id=sid,
                batch_size=self.batch_size_bvp,
                feature_list=BVP_FEATURES,
                df=self.df_w1
            )
            bvp_encoder, bvp_autoenc = self.build_autoencoder_bvp(
                batch_size=self.batch_size_bvp,
                num_features=len(BVP_FEATURES)
            )
            bvp_autoenc.compile(optimizer=RMSprop(learning_rate=0.00025), loss="mse")
            bvp_autoenc.fit(X_train_bvp, X_train_bvp, epochs=4, verbose=1)

            # --------------------------
            # EDA + TEMP
            # --------------------------
            X_train_eda_temp, _, X_test_eda_temp, _, _, _ = self.get_data(
                test_id=sid,
                batch_size=self.batch_size_eda,
                feature_list=EDA_TEMP_FEATURES,
                df=self.df_w2
            )
            eda_temp_encoder, eda_temp_autoenc = self.build_autoencoder_eda_temp(
                batch_size=self.batch_size_eda,
                num_features=len(EDA_TEMP_FEATURES)
            )
            eda_temp_autoenc.compile(optimizer=RMSprop(learning_rate=0.00025), loss="mse")
            eda_temp_autoenc.fit(X_train_eda_temp, X_train_eda_temp, epochs=4, verbose=1)

            # --------------------------
            # Inference (Encoding)
            # --------------------------
            encoded_train_chest = chest_encoder.predict(X_train_chest)
            encoded_test_chest = chest_encoder.predict(X_test_chest)

            encoded_train_bvp = bvp_encoder.predict(X_train_bvp)
            encoded_test_bvp = bvp_encoder.predict(X_test_bvp)

            encoded_train_eda_temp = eda_temp_encoder.predict(X_train_eda_temp)
            encoded_test_eda_temp = eda_temp_encoder.predict(X_test_eda_temp)

            print("encoded_train_chest.shape:", encoded_train_chest.shape)
            print("encoded_train_bvp.shape:  ", encoded_train_bvp.shape)
            print("encoded_train_eda_temp.shape:", encoded_train_eda_temp.shape)

            # --------------------------
            # Truncate to smallest set
            # (so that we can concatenate embeddings consistently)
            # --------------------------
            min_train_size = min(
                encoded_train_chest.shape[0],
                encoded_train_bvp.shape[0],
                encoded_train_eda_temp.shape[0]
            )
            min_test_size = min(
                encoded_test_chest.shape[0],
                encoded_test_bvp.shape[0],
                encoded_test_eda_temp.shape[0]
            )

            # Combine the chest + BVP + EDA/TEMP embeddings, plus the window labels
            combined_train = np.concatenate(
                (
                    encoded_train_chest[:min_train_size],
                    encoded_train_bvp[:min_train_size],
                    encoded_train_eda_temp[:min_train_size],
                    train_labels_int[:min_train_size]
                ),
                axis=1
            )
            combined_test = np.concatenate(
                (
                    encoded_test_chest[:min_test_size],
                    encoded_test_bvp[:min_test_size],
                    encoded_test_eda_temp[:min_test_size],
                    test_labels_int[:min_test_size]
                ),
                axis=1
            )

            # Save to disk
            train_feat_file = f"features/train/feat_loso_{sid}.pkl"
            test_feat_file = f"features/test/feat_loso_{sid}.pkl"
            pd.DataFrame(combined_train).to_pickle(train_feat_file)
            pd.DataFrame(combined_test).to_pickle(test_feat_file)
            
            print(f"Saved combined train features to: {train_feat_file}")
            print(f"Saved combined test features to:  {test_feat_file}")


# -----------------------------------------------------------------------
# Usage
# -----------------------------------------------------------------------
if __name__ == "__main__":
    ae = PhysioAutoencoder()
    ae.train_chest_autoencoders()
    ae.extract_features()
