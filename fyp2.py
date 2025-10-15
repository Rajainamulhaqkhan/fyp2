#!/usr/bin/env python3
"""
dos_detection.py

Train and evaluate models to detect DoS attacks.
Baseline: RandomForest
Neural net: Keras (TensorFlow backend)

Author: Your Name
For final year project — defensive/research use only.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
import matplotlib.pyplot as plt

# Keras
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# -----------------------
# Configuration
# -----------------------
TRAIN_PATH = "data/train.csv"   # change if needed
TEST_PATH  = "data/test.csv"    # optional; if not present, we'll split train
RANDOM_STATE = 42
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# -----------------------
# Utility functions
# -----------------------
def load_csv_auto(path):
    """Try to read CSV; raise helpful error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}. Please place dataset at this path.")
    return pd.read_csv(path)

def identify_feature_types(df, label_col):
    """Return lists: numeric_cols, categorical_cols."""
    X = df.drop(columns=[label_col])
    numeric = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return numeric, categorical

# -----------------------
# Load data
# -----------------------
# Expecting last column is the label; change label_col if different
df_train = load_csv_auto(TRAIN_PATH)
if os.path.exists(TEST_PATH):
    df_test = load_csv_auto(TEST_PATH)
    use_separate_test = True
else:
    df_test = None
    use_separate_test = False

# If label column isn't obvious, assume last column
label_col = df_train.columns[-1]
print(f"Using label column: {label_col}")

# For convenience, unify label values: map multiple attack types into 'attack' vs 'normal' (optional)
# If you want multi-class classification, skip this mapping.
def binarize_label(s):
    # Common NSL-KDD uses label 'normal' for benign; everything else are attack types.
    return np.where(s == "normal", "normal", "attack")

df_train[label_col] = binarize_label(df_train[label_col].astype(str))
if df_test is not None:
    df_test[label_col] = binarize_label(df_test[label_col].astype(str))

# -----------------------
# Identify features
# -----------------------
numeric_cols, categorical_cols = identify_feature_types(df_train, label_col)
print("Numeric cols:", numeric_cols)
print("Categorical cols:", categorical_cols)

# -----------------------
# Preprocessing pipeline
# -----------------------
# We'll one-hot encode categorical columns and scale numeric columns.
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)
    ],
    remainder='drop'
)

# Prepare X, y
X_train_full = df_train.drop(columns=[label_col])
y_train_full = df_train[label_col].values

if use_separate_test:
    X_test = df_test.drop(columns=[label_col])
    y_test = df_test[label_col].values
else:
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
    )

# Fit preprocessor on training data and transform both
print("Fitting preprocessing pipeline...")
preprocessor.fit(X_train_full)
X_train = preprocessor.transform(X_train_full)
X_test  = preprocessor.transform(X_test)

print("Preprocessed feature shape (train):", X_train.shape)
print("Preprocessed feature shape (test):", X_test.shape)

# Optional: handle class imbalance using SMOTE (synthetic oversampling)
print("Before resampling:", pd.Series(y_train_full).value_counts().to_dict())
sm = SMOTE(random_state=RANDOM_STATE, n_jobs=-1)
X_train_bal, y_train_bal = sm.fit_resample(X_train, y_train_full)
print("After resampling:", pd.Series(y_train_bal).value_counts().to_dict())

# -----------------------
# Model 1: Random Forest baseline
# -----------------------
rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
print("Training RandomForest...")
rf.fit(X_train_bal, y_train_bal)

# Evaluate RF
y_pred_rf = rf.predict(X_test)
print("\n=== RandomForest Evaluation ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))
print("Precision:", precision_score(y_test, y_pred_rf, pos_label="attack"))
print("Recall:", recall_score(y_test, y_pred_rf, pos_label="attack"))
print("F1:", f1_score(y_test, y_pred_rf, pos_label="attack"))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("Classification report:\n", classification_report(y_test, y_pred_rf))

joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest_dos.joblib"))
print("Saved RandomForest to", os.path.join(MODEL_DIR, "random_forest_dos.joblib"))

# -----------------------
# Model 2: Keras Neural Network
# -----------------------
input_dim = X_train_bal.shape[1]
print("Keras model input dim:", input_dim)

# Encode labels to 0/1 for Keras
le = LabelEncoder()
y_train_keras = le.fit_transform(y_train_bal)  # normal=0, attack=1 (depends on mapping)
y_test_keras = le.transform(y_test)  # use same encoder

# Simple feedforward network
def build_model(input_dim, lr=1e-3):
    model = models.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

keras_model = build_model(input_dim)
keras_callback = callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("Training Keras model...")
history = keras_model.fit(
    X_train_bal, y_train_keras,
    validation_split=0.15,
    epochs=50,
    batch_size=256,
    callbacks=[keras_callback],
    verbose=2
)

# Evaluate Keras model
y_pred_prob = keras_model.predict(X_test).ravel()
y_pred_keras = (y_pred_prob >= 0.5).astype(int)

print("\n=== Keras NN Evaluation ===")
print("Accuracy:", accuracy_score(y_test_keras, y_pred_keras))
print("Precision:", precision_score(y_test_keras, y_pred_keras))
print("Recall:", recall_score(y_test_keras, y_pred_keras))
print("F1:", f1_score(y_test_keras, y_pred_keras))
print("Confusion matrix:\n", confusion_matrix(y_test_keras, y_pred_keras))
print("Classification report:\n", classification_report(y_test_keras, y_pred_keras, target_names=le.classes_))

keras_model.save(os.path.join(MODEL_DIR, "keras_dos_model.h5"))
joblib.dump(preprocessor, os.path.join(MODEL_DIR, "preprocessor.joblib"))
joblib.dump(le, os.path.join(MODEL_DIR, "label_encoder.joblib"))
print("Saved Keras model and preprocessing artifacts to", MODEL_DIR)

# -----------------------
# Plot training history (loss & accuracy)
# -----------------------
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('Loss')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, "training_history.png"))
print("Saved training history plot to", os.path.join(MODEL_DIR, "training_history.png"))

# -----------------------
# Quick inference example function
# -----------------------
def predict_single(raw_row: pd.Series):
    """
    raw_row: a pd.Series or dict with same columns as training X (before preprocessing)
    returns: label and probability
    """
    if not isinstance(raw_row, pd.Series):
        raw_row = pd.Series(raw_row)
    x = preprocessor.transform(raw_row.to_frame().T)
    p_rf = rf.predict_proba(x)[0][1]  # prob of attack (assuming label order)
    p_nn = keras_model.predict(x).ravel()[0]
    # combine or return both
    return {
        "rf_prob_attack": float(p_rf),
        "nn_prob_attack": float(p_nn),
        "rf_label": "attack" if p_rf >= 0.5 else "normal",
        "nn_label": "attack" if p_nn >= 0.5 else "normal"
    }

# Example usage:
# sample = X_train_full.iloc[0]   # raw sample before preprocessing
# print(predict_single(sample))

print("Done.")
