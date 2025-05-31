"""
Preprocessing Module
===============================================================
including KNN imputation, polynomial features, robust scaling, and feature selection.
"""

import pandas as pd
import numpy as np
import pickle   #is a python std library so fine
import os
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings('ignore')

def preprocess_data(train_data_path, train_labels_path, k_features=20, save_components=True):
    """
    1. Remove text column (46)
    2. KNN imputation with 5 neighbors
    3. Create polynomial features (squares of top 5 features)
    4. Remove low variance features
    5. Robust scaling
    6. Feature selection with f_classif
    
    Args:
        train_data_path: Path to training data file
        train_labels_path: Path to training labels file
        k_features: Number of features to select (will be optimized if needed)
        save_components: Whether to save preprocessing components
    
    Returns:
        tuple: (X_processed, y, components_dict)
    """
    
    print("=== PREPROCESSING PIPELINE ===")
    
    # Load the data
    X_train = pd.read_csv(train_data_path, header=None, sep=',')
    y_train = pd.read_csv(train_labels_path, header=None, names=['label'])
    
    print(f"Original data shape: {X_train.shape}")
    
    # Step 1: Remove text column (46) - EXACT SAME AS 0.78 SCRIPT
    print("Step 1: Removing text column (46)")
    X_processed = X_train.drop(columns=[46]).copy()
    print(f"After removing column 46: {X_processed.shape}")
    
    # Step 2: KNN Imputation with 5 neighbors - EXACT SAME AS 0.78 SCRIPT
    print("Step 2: KNN imputation with 5 neighbors")
    knn_imputer = KNNImputer(n_neighbors=5)
    X_processed = pd.DataFrame(knn_imputer.fit_transform(X_processed))
    print(f"After KNN imputation: {X_processed.shape}")
    
    # Step 3: Feature engineering - polynomial features - EXACT SAME AS 0.78 SCRIPT
    print("Step 3: Creating polynomial features (squares of top 5 features)")
    n_features = min(5, X_processed.shape[1])
    original_feature_count = X_processed.shape[1]
    
    for i in range(n_features):
        X_processed[f'feature_{i}_squared'] = X_processed.iloc[:, i] ** 2
    
    # Convert all column names to strings
    X_processed.columns = [str(col) for col in X_processed.columns]
    print(f"After polynomial features: {X_processed.shape}")
    
    # Step 4: Remove low variance features - EXACT SAME AS 0.78 SCRIPT
    print("Step 4: Removing low variance features")
    variance_selector = VarianceThreshold(threshold=0.001)  # Same threshold as 0.78 script
    X_processed = pd.DataFrame(variance_selector.fit_transform(X_processed))
    print(f"After variance filtering: {X_processed.shape}")
    
    # Step 5: Robust scaling - EXACT SAME AS 0.78 SCRIPT
    print("Step 5: Robust scaling")
    robust_scaler = RobustScaler()
    X_scaled = robust_scaler.fit_transform(X_processed)
    X_scaled = pd.DataFrame(X_scaled)
    print(f"After robust scaling: {X_scaled.shape}")
    
    # Prepare target
    y = y_train.values.ravel()
    
    # Step 6: Feature selection optimization - EXACT SAME AS 0.78 SCRIPT
    print("Step 6: Optimizing feature selection")
    
    # Test the same k_values as the 0.78 script
    k_values = [10, 15, 20, 25, 30]
    best_score = 0
    best_k = None
    best_selector = None
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for k in k_values:
        if k >= X_scaled.shape[1]:
            continue
            
        print(f"  Testing k={k} features...")
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Quick evaluation with logistic regression 
        log_reg = LogisticRegression(random_state=42, class_weight='balanced', max_iter=2000)
        scores = cross_val_score(log_reg, X_selected, y, cv=cv, scoring='f1_weighted')
        mean_score = scores.mean()
        
        print(f"    k={k}: F1 = {mean_score:.4f}")
        
        if mean_score > best_score:
            best_score = mean_score
            best_k = k
            best_selector = selector
    
    print(f"Best k_features: {best_k} with F1 score: {best_score:.4f}")
    
    # Apply best feature selection
    X_final = best_selector.transform(X_scaled)
    X_final = pd.DataFrame(X_final)
    
    print(f"Final preprocessed shape: {X_final.shape}")
    
    # Create components dictionary
    components = {
        'knn_imputer': knn_imputer,
        'variance_selector': variance_selector,
        'robust_scaler': robust_scaler,
        'feature_selector': best_selector,
        'k_features': best_k,
        'n_polynomial_features': n_features,
        'original_feature_count': original_feature_count
    }
    
    # Save components if requested
    if save_components:
        save_preprocessing_components(components)
    
    return X_final, y, components

def apply_preprocessing(X_raw, components):
    """
    Apply the same preprocessing pipeline to new data using saved components.
    
    Args:
        X_raw: Raw input data (pandas DataFrame)
        components: Dictionary of saved preprocessing components
    
    Returns:
        X_processed: Preprocessed data ready for model prediction
    """
    
    # Step 1: Remove text column (46)
    if 46 in X_raw.columns:
        X_processed = X_raw.drop(columns=[46]).copy()
    else:
        X_processed = X_raw.copy()
    
    # Step 2: Apply KNN imputation
    X_processed = pd.DataFrame(components['knn_imputer'].transform(X_processed))
    
    # Step 3: Create polynomial features (same as training)
    n_features = components['n_polynomial_features']
    for i in range(n_features):
        if i < X_processed.shape[1]:
            X_processed[f'feature_{i}_squared'] = X_processed.iloc[:, i] ** 2
    
    # Convert column names to strings
    X_processed.columns = [str(col) for col in X_processed.columns]
    
    # Step 4: Apply variance threshold
    X_processed = pd.DataFrame(components['variance_selector'].transform(X_processed))
    
    # Step 5: Apply robust scaling
    X_scaled = components['robust_scaler'].transform(X_processed)
    X_scaled = pd.DataFrame(X_scaled)
    
    # Step 6: Apply feature selection
    X_final = components['feature_selector'].transform(X_scaled)
    X_final = pd.DataFrame(X_final)
    
    return X_final

def save_preprocessing_components(components, save_dir='preprocessing_components'):
    """Save all preprocessing components to disk using pickle."""
    
    # Create directory if it doesn't exist
    Path(save_dir).mkdir(exist_ok=True)
    
    # Save each component
    for name, component in components.items():
        if component is not None:
            filepath = f'{save_dir}/{name}.pkl'
            with open(filepath, 'wb') as f:
                pickle.dump(component, f)
    
    print(f"✅ Saved preprocessing components to {save_dir}/")

def load_preprocessing_components(save_dir='preprocessing_components'):
    """Load all preprocessing components from disk using pickle."""
    
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
        filepath = f'{save_dir}/{filename}'
        if os.path.exists(filepath):
            component_name = filename.replace('.pkl', '')
            try:
                with open(filepath, 'rb') as f:
                    components[component_name] = pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load {filename}: {e}")
    
    return components

if __name__ == "__main__":
    print("\nExample usage:")
    print("from preprocessing import complete_training_pipeline")
    print("model, score, components = complete_training_pipeline('traindata.txt', 'trainlabels.txt')")
