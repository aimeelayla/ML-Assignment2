import numpy as np
import pandas as pd
import pickle
import os

N_CLASSES = 10

def load_preprocessing_components(save_dir='preprocessing_components'):
    components = {}
    component_files = [
        'knn_imputer.pkl',
        'variance_selector.pkl', 
        'robust_scaler.pkl',
        'feature_selector.pkl',
        'k_features.pkl',
        'n_polynomial_features.pkl',
        'original_feature_count.pkl'
    ]
    for filename in component_files:
        filepath = os.path.join(save_dir, filename)
        if os.path.exists(filepath):
            component_name = filename.replace('.pkl', '')
            with open(filepath, 'rb') as f:
                components[component_name] = pickle.load(f)
    return components

def apply_preprocessing(X_raw, components):
    # Step 1: Remove text column (46)
    if 46 in X_raw.columns:
        X_processed = X_raw.drop(columns=[46]).copy()
    else:
        X_processed = X_raw.copy()

    # Step 2: KNN imputation
    X_processed = pd.DataFrame(components['knn_imputer'].transform(X_processed))

    # Step 3: Polynomial features (square top features)
    n_features = components['n_polynomial_features']
    for i in range(n_features):
        if i < X_processed.shape[1]:
            X_processed[f'feature_{i}_squared'] = X_processed.iloc[:, i] ** 2
    X_processed.columns = [str(col) for col in X_processed.columns]

    # Step 4: Variance threshold
    X_processed = pd.DataFrame(components['variance_selector'].transform(X_processed))

    # Step 5: Robust scaling
    X_scaled = components['robust_scaler'].transform(X_processed)
    X_scaled = pd.DataFrame(X_scaled)

    # Step 6: Feature selection
    X_final = components['feature_selector'].transform(X_scaled)
    X_final = pd.DataFrame(X_final)

    return X_final

def main():
    # READ IN TEST DATA
    test_data = pd.read_csv("testdata.txt", header=None)
    n_datapoints = test_data.shape[0]

    # LOAD PREPROCESSING COMPONENTS
    components = load_preprocessing_components()

    # APPLY PREPROCESSING TO TEST DATA
    X_test = apply_preprocessing(test_data, components)

    # LOAD TRAINED MODEL
    with open('trained_model.pkl', 'rb') as f:
        model = pickle.load(f)

    # PREDICT LABELS
    infer_labels = model.predict(X_test)
    infer_labels = pd.DataFrame(infer_labels)

    # VALIDATE & SAVE INFERRED LABELS
    assert isinstance(infer_labels, pd.DataFrame), f"infer_labels should be a DataFrame, got {type(infer_labels)}"
    assert infer_labels.shape == (n_datapoints, 1), f"Shape should be {(n_datapoints, 1)} but is {infer_labels.shape}"
    assert infer_labels.iloc[:, 0].between(0, N_CLASSES-1).all(), "Predicted labels must be between 0 and 9"

    infer_labels.to_csv("predlabels.txt", index=False, header=False)


if __name__ == "__main__":
    main()
