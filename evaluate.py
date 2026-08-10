import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score

from dataset import load_data

class PinEvaluator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        self.HGB_ALL_PARAMS = {'class_weight': 'balanced', 'l2_regularization': 0, 'learning_rate': 0.3, 'max_depth': 1, 'max_iter': 30}
        self.HGB_SINGLE_PARAMS = {'class_weight': 'balanced', 'l2_regularization': 0, 'learning_rate': 0.3, 'max_depth': 1, 'max_iter': 30}
        self.LR_ALL_PARAMS = {'C': 0.01, 'l1_ratio': 0, 'penalty': 'l2', 'solver': 'liblinear'}
        self.SVM_LINEAR_ALL_PARAMS = {'class_weight': 'balanced', 'max_iter': 10000, 'random_state': 42}
        self.SVM_RBF_ALL_PARAMS = {'C': 240, 'degree': 2, 'gamma': 'scale', 'kernel': 'poly', 'shrinking': False}
        self.RF_ALL_PARAMS = {'class_weight': 'balanced', 'max_depth': 2, 'min_samples_split': 2, 'n_estimators': 100, 'random_state': 42}

        self.models = {
            "Logistic Regression": LogisticRegression(**self.LR_ALL_PARAMS),
            "Linear SVM": LinearSVC(**self.SVM_LINEAR_ALL_PARAMS),
            "RBF SVM": SVC(**self.SVM_RBF_ALL_PARAMS),
            "Random Forest": RandomForestClassifier(**self.RF_ALL_PARAMS),
            "Hist. Gradient Boosting": HistGradientBoostingClassifier(**self.HGB_ALL_PARAMS)
        }
        
    def evaluate_model(self, clf, X_train, y_train, X_test, y_test):
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        return bal_acc

    def clean_modis_names(self, series):
        cleaned_index = series.index.str.replace("modis_warped:", "").str.replace("modis:", "")
        series.index = cleaned_index
        return series.groupby(series.index).mean()

    def get_svm_univariate(self, X_train, y_train, X_test, y_test, feature_names):
        results = []
        n_features = X_train.shape[1]
        for i in range(n_features):
            clf = LinearSVC(class_weight='balanced', penalty="l2", loss="squared_hinge", dual='auto', max_iter=10000, random_state=42)
            clf.fit(X_train[:, i:i+1], y_train)
            y_pred = clf.predict(X_test[:, i:i+1])
            results.append(balanced_accuracy_score(y_test, y_pred))
        return pd.Series(results, index=feature_names)

    def get_hgb_univariate(self, X_train, y_train, X_test, y_test, feature_names):
        results = []
        n_features = X_train.shape[1]
        for i in range(n_features):
            clf = HistGradientBoostingClassifier(**self.HGB_SINGLE_PARAMS)
            clf.fit(X_train[:, i:i+1], y_train)
            y_pred = clf.predict(X_test[:, i:i+1])
            results.append(balanced_accuracy_score(y_test, y_pred))
        return pd.Series(results, index=feature_names)

    def run_feature_importance(self):
        print(f"Loading dataset from {self.data_dir} for feature importance...")
        X_train_full, y_train_str, _, feature_names = load_data(self.data_dir, 'train')
        X_test_full, y_test_str, _, _ = load_data(self.data_dir, 'test')

        le = LabelEncoder()
        y_train = le.fit_transform(y_train_str)
        
        y_test = []
        for label in y_test_str:
            if label in le.classes_:
                y_test.append(le.transform([label])[0])
            else:
                y_test.append(-1)
        y_test = np.array(y_test)
        
        valid_idx = y_test != -1
        X_test_valid_full = X_test_full[valid_idx]
        y_test_valid = y_test[valid_idx]
        
        features_to_keep = []
        for i, f in enumerate(feature_names):
            if f.startswith("s1:"):
                if f.endswith("_log10"):
                    features_to_keep.append(i)
            else:
                features_to_keep.append(i)
                
        X_train_full = X_train_full[:, features_to_keep]
        X_test_valid_full = X_test_valid_full[:, features_to_keep]
        feature_names = [f.replace("_log10", "") if f.startswith("s1:") else f for i, f in enumerate(feature_names) if i in features_to_keep]    
        
        unwarped_idx = []
        warped_idx = []
        for i, f in enumerate(feature_names):
            if f.startswith("modis_warped:"):
                warped_idx.append(i)
            elif f.startswith("modis:"):
                unwarped_idx.append(i)
            else:
                unwarped_idx.append(i)
                warped_idx.append(i)
                
        feature_names = np.array(feature_names)
        unwarped_features = feature_names[unwarped_idx]
        warped_features = feature_names[warped_idx]
        
        X_train_unw = X_train_full[:, unwarped_idx]
        X_test_unw = X_test_valid_full[:, unwarped_idx]
        
        X_train_warp = X_train_full[:, warped_idx]
        X_test_warp = X_test_valid_full[:, warped_idx]

        print("Evaluating Unwarped Features...")
        svm_unw = self.get_svm_univariate(X_train_unw, y_train, X_test_unw, y_test_valid, unwarped_features)
        hgb_unw = self.get_hgb_univariate(X_train_unw, y_train, X_test_unw, y_test_valid, unwarped_features)
        
        print("Evaluating Warped Features...")
        svm_warp = self.get_svm_univariate(X_train_warp, y_train, X_test_warp, y_test_valid, warped_features)
        hgb_warp = self.get_hgb_univariate(X_train_warp, y_train, X_test_warp, y_test_valid, warped_features)

        # Base Features Table
        base_mask = svm_unw.index.str.startswith(("s1:", "amsr:", "topo:"))
        df_base = pd.DataFrame({
            'SVM Balanced Accuracy': svm_unw[base_mask],
            'GB Balanced Accuracy': hgb_unw[base_mask]
        }).fillna(0)

        df_base['Sort Key'] = (df_base['SVM Balanced Accuracy'] + df_base['GB Balanced Accuracy']) / 2
        df_base = df_base.sort_values(by='Sort Key', ascending=False).drop(columns=['Sort Key']).head(4)
        
        print("\n--- Base Feature Importance ---")
        print(df_base.to_string(float_format=lambda x: f"{x:.4f}"))

        # MODIS Features Table
        svm_unw_modis = self.clean_modis_names(svm_unw[svm_unw.index.str.startswith("modis:")])
        svm_warp_modis = self.clean_modis_names(svm_warp[svm_warp.index.str.startswith("modis_warped:")])
        hgb_unw_modis = self.clean_modis_names(hgb_unw[hgb_unw.index.str.startswith("modis:")])
        hgb_warp_modis = self.clean_modis_names(hgb_warp[hgb_warp.index.str.startswith("modis_warped:")])

        df_modis = pd.DataFrame({
            'SVM (Unwarped)': svm_unw_modis,
            'SVM (Warped)': svm_warp_modis,
            'GB (Unwarped)': hgb_unw_modis,
            'GB (Warped)': hgb_warp_modis
        }).fillna(0)
        
        df_modis['Sort Key'] = (df_modis['SVM (Warped)'] + df_modis['GB (Warped)']) / 2
        df_modis = df_modis.sort_values(by='Sort Key', ascending=False).drop(columns=['Sort Key']).head(15)
        
        print("\n--- MODIS Feature Importance ---")
        print(df_modis.to_string(float_format=lambda x: f"{x:.4f}"))

        
    def run_aa_comparison(self):
        print(f"\nLoading dataset from {self.data_dir} for model comparison...")
        X_train_unw, y_train_str, _, _ = load_data(self.data_dir, 'train', modis_mode='unwarped')
        X_test_unw_full, y_test_str, y_oracle_str, _ = load_data(self.data_dir, 'test', modis_mode='unwarped')
        
        X_train_warp, _, _, _ = load_data(self.data_dir, 'train', modis_mode='warped')
        X_test_warp_full, _, _, _ = load_data(self.data_dir, 'test', modis_mode='warped')

        le = LabelEncoder()
        y_train = le.fit_transform(y_train_str)
        
        y_test = []
        for label in y_test_str:
            if label in le.classes_:
                y_test.append(le.transform([label])[0])
            else:
                y_test.append(-1)
        y_test = np.array(y_test)
        
        valid_idx = y_test != -1
        X_test_unw = X_test_unw_full[valid_idx]
        X_test_warp = X_test_warp_full[valid_idx]
        y_test_valid = y_test[valid_idx]

        y_oracle = []
        if y_oracle_str is not None:
            for label in y_oracle_str:
                if label in le.classes_:
                    y_oracle.append(le.transform([label])[0])
                else:
                    y_oracle.append(-1)
            y_oracle = np.array(y_oracle)
            y_oracle_valid = y_oracle[valid_idx]
            bal_acc_oracle = balanced_accuracy_score(y_test_valid, y_oracle_valid)
        else:
            bal_acc_oracle = None

        bal_acc_all_0 = balanced_accuracy_score(y_test_valid, np.zeros_like(y_test_valid))

        results = []

        print("\n--- Evaluating Unwarped Features ---")
        for model_name, clf in self.models.items():
            print(f"Evaluating {model_name} (Unwarped)...")
            bal_acc = self.evaluate_model(clf, X_train_unw, y_train, X_test_unw, y_test_valid)
            results.append({
                "Model": model_name,
                "Dataset": "Unwarped",
                "Balanced Accuracy": bal_acc
            })

        print("\n--- Evaluating Warped Features ---")
        for model_name, clf in self.models.items():
            print(f"Evaluating {model_name} (Warped)...")
            bal_acc = self.evaluate_model(clf, X_train_warp, y_train, X_test_warp, y_test_valid)
            results.append({
                "Model": model_name,
                "Dataset": "Warped",
                "Balanced Accuracy": bal_acc
            })

        df = pd.DataFrame(results)
        pivot_df = df.pivot(index="Model", columns="Dataset", values="Balanced Accuracy")
        
        if "Unwarped" in pivot_df.columns and "Warped" in pivot_df.columns:
            pivot_df = pivot_df[["Unwarped", "Warped"]]
            pivot_df.columns = ["Balanced Accuracy (Unwarped)", "Balanced Accuracy (Warped)"]

        pivot_df = pivot_df.sort_values(by="Balanced Accuracy (Warped)", ascending=False)

        pivot_df.loc['All Ice Baseline'] = [bal_acc_all_0, bal_acc_all_0]
        if bal_acc_oracle is not None:
            pivot_df.loc['Oracle Baseline'] = [bal_acc_oracle, bal_acc_oracle]
        
        print("\n--- Model Comparison ---")
        print(pivot_df.to_string(float_format=lambda x: f"{x:.4f}"))

    def run_all(self):
        self.run_feature_importance()
        self.run_aa_comparison()

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "dataset")
    evaluator = PinEvaluator(data_dir)
    evaluator.run_all()
