from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def preprocess_data(train_data_path, train_labels_path, save_components=True):
    print("=== PREPROCESSING PIPELINE WITH RANDOM FOREST ===")

    X_train = pd.read_csv(train_data_path, header=None, sep=',')
    y_train = pd.read_csv(train_labels_path, header=None, names=['label'])
    print(f"Original data shape: {X_train.shape}")

    # Step 1: Drop text column (column 46)
    print("Step 1: Removing text column (46)")
    X_processed = X_train.drop(columns=[46]).copy()

    # Step 2: KNN Imputation
    print("Step 2: KNN imputation")
    knn_imputer = KNNImputer(n_neighbors=5)
    X_processed = pd.DataFrame(knn_imputer.fit_transform(X_processed))

    # Step 3: Polynomial features (squares + interaction terms for top 5)
    print("Step 3: Adding polynomial features (squares + interaction terms)")
    top_n = min(5, X_processed.shape[1])
    base_features = X_processed.iloc[:, :top_n].copy()

    for i in range(top_n):
        X_processed[f'feature_{i}_squared'] = base_features.iloc[:, i] ** 2

    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    interaction_features = poly.fit_transform(base_features)
    interaction_only = interaction_features[:, top_n:]  # exclude original features

    for idx in range(interaction_only.shape[1]):
        X_processed[f'interaction_{idx}'] = interaction_only[:, idx]

    # Step 4: Remove low-variance features
    print("Step 4: Removing low variance features")
    X_processed.columns = X_processed.columns.astype(str)
    variance_selector = VarianceThreshold(threshold=0.001)
    X_processed = pd.DataFrame(variance_selector.fit_transform(X_processed))

    # Step 5: Robust scaling
    print("Step 5: Robust scaling")
    robust_scaler = RobustScaler()
    X_scaled = pd.DataFrame(robust_scaler.fit_transform(X_processed))

    y = y_train.values.ravel()

    # Step 6: Feature selection and Random Forest hyperparameter tuning
    print("Step 6: Feature selection and Random Forest tuning")
    k_values = [10, 15, 20, 25, 30]
    best_score = 0
    best_k = None
    best_selector = None
    best_model = None

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for k in k_values:
        if k >= X_scaled.shape[1]:
            continue

        print(f"  Testing k={k} features...")
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_scaled, y)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced'],
            'random_state': [42]
        }

        rf_clf = RandomForestClassifier()
        grid_search = GridSearchCV(rf_clf, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
        grid_search.fit(X_selected, y)

        mean_score = grid_search.best_score_
        print(f"    k={k}: Best F1 = {mean_score:.4f} with params {grid_search.best_params_}")

        if mean_score > best_score:
            best_score = mean_score
            best_k = k
            best_selector = selector
            best_model = grid_search.best_estimator_

    print(f"Best k_features: {best_k} with F1 score: {best_score:.4f}")

    # Retrain final model
    X_final = best_selector.transform(X_scaled)
    best_model.fit(X_final, y)

    components = {
        'knn_imputer': knn_imputer,
        'variance_selector': variance_selector,
        'robust_scaler': robust_scaler,
        'feature_selector': best_selector,
        'random_forest_model': best_model,
        'k_features': best_k,
        'interaction_terms': poly,
        'n_polynomial_features': top_n
    }

    if save_components:
        save_preprocessing_components(components)

    with open("trained_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print("🎉 Model training and saving complete.")
    return best_model, best_score, components

def save_preprocessing_components(components, save_dir='preprocessing_components'):
    Path(save_dir).mkdir(exist_ok=True)
    for name, component in components.items():
        if component is not None:
            with open(f"{save_dir}/{name}.pkl", "wb") as f:
                pickle.dump(component, f)
    print(f"✅ Saved preprocessing components to {save_dir}/")

if __name__ == "__main__":
    model, best_f1, components = preprocess_data("traindata.txt", "trainlabels.txt")
    print(f"Final best F1 score on training CV: {best_f1:.4f}")
