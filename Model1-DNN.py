import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
import optuna
import matplotlib.pyplot as plt

#import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Then load your data, build the model, run training

# -------------------------------------------------------------------------
# 1) Load Data & Filter
#    NOTE: merged_chest.pkl typically contains columns:
#    ['sid', 'acc1', 'acc2', 'acc3', 'ecg', 'emg', 'eda', 'temp', 'resp', 'label']
# -------------------------------------------------------------------------
df = pd.read_pickle('data/merged_chest.pkl')
print("Columns in merged_chest.pkl:", df.columns.tolist())
# For chest data, we have 'temp' NOT 'c_temp'.

# Filter rows: only keep valid temperatures and label ∈ [1,2,3].
df = df[df["temp"] > 0]
df = df[df["label"].isin([1, 2, 3])]

# Choose a relevant feature list from the chest columns you actually have
# Example: ecg, emg, eda, temp, resp
feat_list = ["ecg", "emg", "eda", "temp", "resp"]
print("Using features:", feat_list)

# Unique subject IDs
ids = df["sid"].unique().astype(int)
print("Subject IDs:", ids)

# -------------------------------------------------------------------------
# 2) Build ANN Model
# -------------------------------------------------------------------------
def ANN_model(n_input_dim, n_out_dim, num_hidden1, num_hidden2, drop_rate, weight_decay):
    """
    Build a simple 2-hidden-layer feedforward network.
    """
    model = Sequential()
    # 1st hidden layer
    model.add(Dense(num_hidden1, input_dim=n_input_dim, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Dropout(drop_rate))
    
    # 2nd hidden layer
    model.add(Dense(num_hidden2, activation='relu',
                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay)))
    model.add(Dropout(drop_rate))
    
    # Output layer (softmax for classification)
    model.add(Dense(n_out_dim, activation='softmax'))
    return model

def one_hot_enc(labels, num_classes):
    """
    Convert integer labels (1,2,3,...) to one-hot vectors.
    Example: if num_classes=3 and labels=[1,2,3,2], returns shape (4,3).
    """
    enc = np.zeros((labels.shape[0], num_classes))
    for i, val in enumerate(labels):
        # val-1 because labels are 1-based
        enc[i, val-1] = 1
    return enc

# -------------------------------------------------------------------------
# 3) Objective Function for Optuna
#    Leave-One-Subject-Out CV to get average test accuracy
# -------------------------------------------------------------------------
def objective(trial):
    # Hyperparameters to tune
    num_hidden1   = trial.suggest_int("n_units_l0", 8, 10, log=True)
    num_hidden2   = trial.suggest_int("n_units_l1", 4, 8, log=True)
    drop_rate     = trial.suggest_float("drop_rate", 0.2, 0.4)
    weight_decay  = trial.suggest_float("weight_decay", 1e-8, 1e-5, log=True)
    lr            = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size    = trial.suggest_int("batch_size", 32, 128)
    epochs        = trial.suggest_int("epochs", 5, 15)

    test_metrics = []
    
    # Leave-one-subject-out
    for sid in ids:
        # Train on all other subjects
        df_train = df[df["sid"] != sid]
        df_test = df[df["sid"] == sid]
        
        # Scale features
        x_train = StandardScaler().fit_transform(df_train[feat_list])
        x_test = StandardScaler().fit_transform(df_test[feat_list])
        
        y_train = df_train["label"].values.astype(int)
        y_test = df_test["label"].values.astype(int)
        
        # Number of classes
        K = len(np.unique(y_train))
        
        # One-hot encode labels
        y_train_oh = one_hot_enc(y_train, K)
        y_test_oh  = one_hot_enc(y_test, K)
        
        # Build and compile the model
        model = ANN_model(x_train.shape[1], K,
                          num_hidden1, num_hidden2,
                          drop_rate, weight_decay)
        
        optimizer = SGD(learning_rate=lr)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        
        # Train
        model.fit(x_train, y_train_oh, epochs=epochs, batch_size=batch_size, verbose=1,
                  validation_data=(x_test, y_test_oh))
        
        # Evaluate
        loss, acc = model.evaluate(x_test, y_test_oh, verbose=1)
        print(f"[Subject {sid}] Loss={loss:.3f}, Acc={acc:.3f}")
        test_metrics.append(acc)
    
    # Return the average test accuracy
    return np.mean(test_metrics)

# -------------------------------------------------------------------------
# 4) Run Optuna Study (Comment out if you don't want hyperparameter tuning)
# -------------------------------------------------------------------------
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=10)


print("\n== Optuna Results ==")
print(f"Number of finished trials: {len(study.trials)}")
print("Best trial:")
print(f"  Value (Accuracy): {study.best_trial.value:.3f}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

# -------------------------------------------------------------------------
# 5) Optional: Train a final model with the best hyperparams
# -------------------------------------------------------------------------
best_params = study.best_trial.params

# If you prefer fixed values, or if you want to override:
'''best_params = {
     "n_units_l0": 8,
     "n_units_l1": 8,
     "drop_rate": 0.3,
     "weight_decay": 1e-6,
     "lr": 1e-3,
     "batch_size": 64,
     "epochs": 10
}
'''
# -------------------------------------------------------------------------
# 6) Final Training/Evaluation with LOO (including f1-score)
# -------------------------------------------------------------------------
def train_model(params):
    test_metrics = []
    
    num_hidden1   = params["n_units_l0"]
    num_hidden2   = params["n_units_l1"]
    drop_rate     = params["drop_rate"]
    weight_decay  = params["weight_decay"]
    lr            = params["lr"]
    batch_size    = params["batch_size"]
    epochs        = params["epochs"]
    
    for sid in ids:
        df_train = df[df["sid"] != sid]
        df_test  = df[df["sid"] == sid]
        
        # Standardize features
        x_train = StandardScaler().fit_transform(df_train[feat_list])
        x_test  = StandardScaler().fit_transform(df_test[feat_list])
        
        y_train = df_train["label"].values.astype(int)
        y_test  = df_test["label"].values.astype(int)
        
        K = len(np.unique(y_train))
        y_train_oh = one_hot_enc(y_train, K)
        y_test_oh  = one_hot_enc(y_test, K)
        
        model = ANN_model(x_train.shape[1], K,
                          num_hidden1, num_hidden2,
                          drop_rate, weight_decay)
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(learning_rate=lr),
                      metrics=['accuracy'])
        
        model.fit(x_train, y_train_oh,
                  epochs=epochs,
                  batch_size=batch_size,
                  verbose=1)
        
        loss, acc = model.evaluate(x_test, y_test_oh, verbose=1)
        
        # Predict to compute F1-score
        y_pred_probs = model.predict(x_test, verbose=1)
        y_pred = np.argmax(y_pred_probs, axis=1)
        # Weighted F1
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"[Subject {sid}] Acc={acc:.3f} | F1={f1:.3f}")
        test_metrics.append([sid, acc, f1])
    
    return np.array(test_metrics)

print("\n== Training final model with best hyperparams ==")
scores = train_model(best_params)

acc_mean = scores[:,1].mean()
acc_std = scores[:,1].std()
f1_mean = scores[:,2].mean()
f1_std = scores[:,2].std()

print("\n=== Final Results (LOO) ===")
print(f"Accuracy: {acc_mean:.3f} ± {acc_std:.3f}")
print(f"F1-score: {f1_mean:.3f} ± {f1_std:.3f}")

# -------------------------------------------------------------------------
# 7) (Optional) Plot subject-wise F1-scores
# -------------------------------------------------------------------------
subject_ids = scores[:,0].astype(int)
plt.figure()
plt.bar(subject_ids, scores[:,2]*100)
plt.xticks(subject_ids)
plt.ylim([0, 100])
plt.xlabel("Subject ID")
plt.ylabel("F1-score (%)")
plt.title("Subject-wise F1-scores (LOO)")
plt.show()
