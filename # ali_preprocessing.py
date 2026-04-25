# ali_preprocessing.py
# Author: Ali Hussam
# Project: ML Project - Team 6
# Task: Outliers Handling, Feature Transformation, Dimensionality Reduction, and Imbalance (SMOTE/Undersampling)
# This module works after Sayed's preprocessing (Encoding, Missing Values, Normalization)

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


class DataProcessor:
    """
    Advanced Data Preprocessing Module
    Scope: Outlier Handling, Feature Transformation, Selection, and Imbalance.
    """
    
    def __init__(self, dataframe):
        self.df = dataframe.copy()

    # --- SECTION 1: OUTLIER DETECTION & HANDLING ---
    
    def apply_iqr_filter(self, column, strategy='remove'):
        """Interquartile Range (IQR) Method for Outliers"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_limit = Q1 - 1.5 * IQR
        upper_limit = Q3 + 1.5 * IQR
        
        outlier_mask = (self.df[column] < lower_limit) | (self.df[column] > upper_limit)
        
        if strategy == 'remove':
            self.df = self.df[~outlier_mask]
        elif strategy == 'clip':
            self.df[column] = np.clip(self.df[column], lower_limit, upper_limit)
            
        return outlier_mask.sum()

    def apply_zscore_filter(self, column, threshold=3):
        """Z-Score Standard Deviation Method"""
        z_scores = np.abs(stats.zscore(self.df[column].fillna(self.df[column].mean())))
        outlier_mask = z_scores > threshold
        self.df = self.df[~outlier_mask]
        return outlier_mask.sum()

    def apply_winsorization(self, column, limits=[0.05, 0.05]):
        """Data Winsorization (Percentile Capping)"""
        self.df[column] = stats.mstats.winsorize(self.df[column], limits=limits)
        return True

    # --- SECTION 2: FEATURE TRANSFORMATION ---

    def map_feature_distribution(self, columns, method='log'):
        """Log, Box-Cox, and Power Transformations"""
        if method == 'log':
            self.df[columns] = np.log1p(self.df[columns])
        else:
            transformer = PowerTransformer(method=method)
            self.df[columns] = transformer.fit_transform(self.df[columns])
        return True

    def generate_polynomial_features(self, columns, degree=2):
        """Non-linear Feature Generation"""
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        poly_data = poly.fit_transform(self.df[columns])
        poly_cols = poly.get_feature_names_out(columns)
        
        poly_df = pd.DataFrame(poly_data, columns=poly_cols, index=self.df.index)
        self.df = pd.concat([self.df.drop(columns=columns), poly_df], axis=1)
        return True

    # --- SECTION 3: DIMENSIONALITY REDUCTION & SELECTION ---

    def execute_rfe_selection(self, target_label, n_features=5):
        """Recursive Feature Elimination (RFE)"""
        X = self.df.drop(columns=[target_label])
        y = self.df[target_label]
        
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFE(estimator, n_features_to_select=n_features)
        selector = selector.fit(X, y)
        
        selected_features = X.columns[selector.support_].tolist()
        self.df = self.df[selected_features + [target_label]]
        return selected_features

    def execute_pca(self, n_components=2):
        """Principal Component Analysis (PCA)"""
        numeric_data = self.df.select_dtypes(include=[np.number])
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(numeric_data)
        
        self.df = pd.DataFrame(
            transformed_data, 
            columns=[f'PC{i+1}' for i in range(transformed_data.shape[1])],
            index=self.df.index
        )
        return np.sum(pca.explained_variance_ratio_)

    # --- SECTION 4: CLASS IMBALANCE HANDLING ---

    def resample_data(self, target_label, technique='smote'):
        """Oversampling (SMOTE) and Undersampling Techniques"""
        X = self.df.drop(columns=[target_label])
        y = self.df[target_label]
        
        sampler = SMOTE(random_state=42) if technique == 'smote' else RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        
        self.df = pd.concat([pd.DataFrame(X_res, columns=X.columns), 
                             pd.Series(y_res, name=target_label)], axis=1)
        return True

    def export_processed_data(self):
        """Returns the final cleaned DataFrame"""
        return self.df