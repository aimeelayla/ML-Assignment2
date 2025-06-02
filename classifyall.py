import numpy as np
import pandas as pd
import pickle
import os

N_CLASSES = 10

def load_preprocessing_components(save_dir='preprocessing_components'):
    components = {}
    filenames = [
        'knn_imputer.pkl',
        'variance_selector.pkl',
        'robust_scaler.pkl',
        'feature_selector.pkl',
        'interaction_terms.pkl',
        'n_polynomial_features.pkl'
    ]
    for fname in filenames:
        path = os.path.join(save_dir, fname)
        if os.path.exists(path):
            key = fname.replace('.pkl', '')
            with open(path, 'rb') as f:
                components[key] = pickle.load(f)
    return components

def apply_preprocessing(X_raw, components):
    # Step 1: Drop column 46 if present
    if 46 in X_raw.columns:
        X_processed = X_raw.drop(columns=[46]).copy()
    else:
        X_processed = X_raw.copy()

    # Step 2: Impute missing values
    X_processed = pd.DataFrame(components['knn_imputer'].transform(X_processed))

    # Step 3: Polynomial features (squares + interactions for top-N features)
    n = components['n_polynomial_features']
    base = X_processed.iloc[:, :n].copy()

    # Add squared terms
    for i in range(n):
        X_processed[f'feature_{i}_squared'] = base.iloc[:, i] ** 2

    # Add interaction terms
    poly = components['interaction_terms']
    interaction_features = poly.transform(base)
    interaction_only = interaction_features[:, n:]  # remove original features

    for idx in range(interaction_only.shape[1]):
        X_processed[f'interaction_{idx}'] = interaction_only[:, idx]

    # Ensure all column names are strings
    X_processed.columns = X_processed.columns.astype(str)

    # Step 4: Remove low variance features
    X_processed = pd.DataFrame(components['variance_selector'].transform(X_processed))

    # Step 5: Robust scaling
    X_scaled = pd.DataFrame(components['robust_scaler'].transform(X_processed))

    # Step 6: Feature selection
    X_final = pd.DataFrame(components['feature_selector'].transform(X_scaled))

    return X_final

def main():
    # Load raw test data
    X_raw = pd.read_csv("testdata.txt", header=None)

    # Load preprocessing components
    components = load_preprocessing_components()

    # Apply transformations
    X_processed = apply_preprocessing(X_raw, components)

    # Load trained model
    with open("trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predict labels
    y_pred = model.predict(X_processed)
    y_pred = pd.DataFrame(y_pred)

    # Sanity checks
    assert y_pred.shape == (X_raw.shape[0], 1)
    assert y_pred.iloc[:, 0].between(0, N_CLASSES - 1).all()

    # Save to file
    y_pred.to_csv("predlabels.txt", index=False, header=False)

if __name__ == "__main__":
    main()
