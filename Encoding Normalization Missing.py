import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler 
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

class PreprocessingHandler:
    
    def __init__(self, df):
        self.df = df.copy()  # copy original data
        self.encoders = {}    # inverse transformation
        
    # ==================== METHODS ====================
       # Validate and convert column input to list
    def _validate_and_prepare_columns(self, column_names):
        
        if isinstance(column_names, str):
            column_names = [column_names]
        
        # Check columns exist
        missing_cols = [col for col in column_names if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        return column_names
    
    #Return columns that contain missing values
    def _get_columns_with_missing(self, column_names):
       
        return [col for col in column_names if self.df[col].isnull().any()]
    # Check and report missing values
    def _check_and_report_missing(self, column_names):
        cols_with_missing = self._get_columns_with_missing(column_names)
        if not cols_with_missing:
            print(f"No missing values found in columns: {column_names}")
            return None
        return cols_with_missing
     # fit and transform the scaler
    def _apply_fit_transform_and_store(self, column_names, transformer, storage_key_prefix):
        self.df[column_names] = transformer.fit_transform(self.df[column_names])
        self.encoders[f'{storage_key_prefix}_{tuple(column_names)}'] = transformer
        return transformer
    
    # ==================== DATA ENCODING ====================
    
    def label_encode(self, column_name):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        # missing values in the column
        if self.df[column_name].isnull().any():
            # fill nan values 
            self.df[column_name] = self.df[column_name].fillna("Missing")
            print(f"Filled missing values with placeholder 'Missing'")
        
        # Convert column to string
        self.df[column_name] = self.df[column_name].astype(str)
        
        # Initialize and fit the label encoder
        label_encoder = LabelEncoder()
        self.df[column_name] = label_encoder.fit_transform(self.df[column_name])
        
        # Store the encoder 
        self.encoders[f'label_{column_name}'] = label_encoder
        
        print(f"Label encoding applied to column '{column_name}'")
        print(f"Unique encoded values: {sorted(self.df[column_name].unique())}")
        
        return self.df
    
    def one_hot_encode(self, column_name, drop_first=False):
        if column_name not in self.df.columns:
            raise ValueError(f"Column '{column_name}' not found in dataframe")
        
        # Store original column 
        original_column = self.df[column_name].copy()
        
        # one-hot encoding
        one_hot_encoded = pd.get_dummies(
            self.df[column_name], 
            prefix=column_name, 
            drop_first=drop_first,
            dtype=int
        )
        
        # concat the encode columns
        self.df = pd.concat([self.df.drop(columns=[column_name]), one_hot_encoded], axis=1)
        
        # Store encoding
        self.encoders[f'onehot_{column_name}'] = {
            'original_values': original_column,
            'new_columns': list(one_hot_encoded.columns),
            'drop_first': drop_first
        }
        
        print(f"One-hot encoding applied to column '{column_name}'")
        print(f"Created {len(one_hot_encoded.columns)} new columns: {list(one_hot_encoded.columns)}")
        
        return self.df
    
    # ==================== NORMALIZATION and SCALING ====================    
    
    def standard_scale(self, column_names):
        column_names = self._validate_and_prepare_columns(column_names)
        
        scaler = StandardScaler()
        self._apply_fit_transform_and_store(column_names, scaler, 'standard_scaler')
        
        print(f"Standard scaling applied to columns: {column_names}")
        print(f"Each column now has mean ≈ 0 and std ≈ 1")
        
        return self.df
    
    def minmax_scale(self, column_names, feature_range=(0, 1)):
        column_names = self._validate_and_prepare_columns(column_names)
        
        scaler = MinMaxScaler(feature_range=feature_range)
        self._apply_fit_transform_and_store(column_names, scaler, 'minmax_scaler')
        
        print(f"MinMax scaling applied to columns: {column_names}")
        print(f"Data scaled to range {feature_range}")
        
        return self.df
    
    # ==================== HANDLING MISSING VALUES ====================
    
    def simple_impute(self, column_names, strategy='mean'):
        column_names = self._validate_and_prepare_columns(column_names)
        
        cols_with_missing = self._check_and_report_missing(column_names)
        if cols_with_missing is None:
            return self.df
        
        imputer = SimpleImputer(strategy=strategy)
        self._apply_fit_transform_and_store(column_names, imputer, 'simple_imputer')
        
        print(f"Simple imputation applied to columns: {cols_with_missing}")
        print(f"Strategy used: {strategy}")
        
        # Report imputation statistics
        for col in cols_with_missing:
            null_count_before = self.df[col].isnull().sum() if col in self.df.columns else 0
            print(f"{col}: {null_count_before} missing values imputed")
        
        return self.df
    
    def knn_impute(self, column_names, n_neighbors=5, weights='uniform'):
        column_names = self._validate_and_prepare_columns(column_names)
        
        cols_with_missing = self._check_and_report_missing(column_names)
        if cols_with_missing is None:
            return self.df
        
        # KNN imputer
        imputer = KNNImputer(n_neighbors=n_neighbors, weights=weights)
        self._apply_fit_transform_and_store(column_names, imputer, 'knn_imputer')
        
        print(f"KNN imputation applied to columns: {cols_with_missing}")
        print(f"Parameters: n_neighbors={n_neighbors}, weights='{weights}'")
        
        return self.df
    
    def iterative_impute(self, column_names, max_iter=10, random_state=None):
        column_names = self._validate_and_prepare_columns(column_names)
        
        cols_with_missing = self._check_and_report_missing(column_names)
        if cols_with_missing is None:
            return self.df
        
        # Initialize iterative imputer
        imputer = IterativeImputer(max_iter=max_iter, random_state=random_state)
        self._apply_fit_transform_and_store(column_names, imputer, 'iterative_imputer')
        
        print(f"Iterative imputation applied to columns: {cols_with_missing}")
        print(f"Parameters: max_iter={max_iter}, random_state={random_state}")
        
        return self.df
    
    def get_dataframe(self):
        return self.df 