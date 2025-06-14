import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from datetime import datetime
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

warnings.filterwarnings('ignore')

class DataProcessor:
    """Advanced data processing cl# Initialize the processor
    processor = DataProcessor()
    
    # Load data
    df = processor.load_data('data/insurance_data.csv')
    
    # Get data overview
    overview = processor.get_data_overview()
    
    # Clean data
    cleaned_df = processor.clean_data(
        remove_duplicates=True,
        handle_missing='auto',
        outlier_method='iqr',
        outlier_threshold=1.5
    )
    
    # Create features
    feature_df = processor.create_insurance_features()
    
    # Encode categorical variables
    encoded_df = processor.encode_categorical_features(method='onehot')ass for insurance analytics."""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.logger = self._setup_logger()
        self.data = None
        self.processed_data = None
        self.feature_mappings = {}
        self.encoders = {}
        self.scalers = {}
        
    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def load_data(self, file_path: str, **kwargs) -> pd.DataFrame:
        """Load insurance data from various file formats."""
        try:
            if file_path.endswith('.csv'):
                self.data = pd.read_csv(file_path, **kwargs)
            elif file_path.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_path, **kwargs)
            elif file_path.endswith('.parquet'):
                self.data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
                
            self.logger.info(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def get_data_overview(self) -> Dict:
        """Generate comprehensive data overview."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        overview = {
            'shape': self.data.shape,
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum(),
            'duplicate_rows': self.data.duplicated().sum()
        }
        
        # Numerical columns statistics
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            overview['numerical_summary'] = self.data[numerical_cols].describe().to_dict()
            
        # Categorical columns statistics
        categorical_cols = self.data.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            overview['categorical_summary'] = {}
            for col in categorical_cols:
                overview['categorical_summary'][col] = {
                    'unique_values': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts().head().to_dict()
                }
        
        return overview
    
    def clean_data(self, 
                   remove_duplicates: bool = True,
                   handle_missing: str = 'auto',
                   outlier_method: str = 'iqr',
                   outlier_threshold: float = 1.5) -> pd.DataFrame:
        """Comprehensive data cleaning pipeline."""
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")
            
        cleaned_data = self.data.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_shape = cleaned_data.shape[0]
            cleaned_data = cleaned_data.drop_duplicates()
            removed_duplicates = initial_shape - cleaned_data.shape[0]
            if removed_duplicates > 0:
                self.logger.info(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        cleaned_data = self._handle_missing_values(cleaned_data, method=handle_missing)
        
        # Handle outliers
        cleaned_data = self._handle_outliers(
            cleaned_data, 
            method=outlier_method, 
            threshold=outlier_threshold
        )
        
        # Data type optimization
        cleaned_data = self._optimize_dtypes(cleaned_data)
        
        self.processed_data = cleaned_data
        self.logger.info(f"Data cleaning completed. Final shape: {cleaned_data.shape}")
        return cleaned_data
    
    def _handle_missing_values(self, data: pd.DataFrame, method: str = 'auto') -> pd.DataFrame:
        """Handle missing values based on specified method."""
        if method == 'auto':
            # Intelligent missing value handling
            for col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                
                if missing_pct > 0.5:
                    # Drop columns with >50% missing values
                    data = data.drop(columns=[col])
                    self.logger.info(f"Dropped column {col} (missing: {missing_pct:.2%})")
                elif missing_pct > 0:
                    if data[col].dtype in ['object']:
                        # Fill categorical with mode
                        data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
                    else:
                        # Fill numerical with median
                        data[col] = data[col].fillna(data[col].median())
                        
        elif method == 'drop':
            data = data.dropna()
        elif method == 'fill':
            # Use KNN imputation for numerical, mode for categorical
            num_cols = data.select_dtypes(include=[np.number]).columns
            cat_cols = data.select_dtypes(include=['object']).columns
            
            if len(num_cols) > 0:
                imputer = KNNImputer(n_neighbors=5)
                data[num_cols] = imputer.fit_transform(data[num_cols])
                
            for col in cat_cols:
                data[col] = data[col].fillna(data[col].mode()[0] if not data[col].mode().empty else 'Unknown')
        
        return data
    
    def _handle_outliers(self, data: pd.DataFrame, 
                        method: str = 'iqr', 
                        threshold: float = 1.5) -> pd.DataFrame:
        """Handle outliers using specified method."""
        numerical_cols = data.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            if method == 'iqr':
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing
                    data.loc[data[col] < lower_bound, col] = lower_bound
                    data.loc[data[col] > upper_bound, col] = upper_bound
                    self.logger.info(f"Capped {outlier_count} outliers in {col}")
                    
            elif method == 'zscore':
                z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                outliers = z_scores > threshold
                outlier_count = outliers.sum()
                
                if outlier_count > 0:
                    # Cap at mean ± threshold * std
                    lower_bound = data[col].mean() - threshold * data[col].std()
                    upper_bound = data[col].mean() + threshold * data[col].std()
                    data.loc[data[col] < lower_bound, col] = lower_bound
                    data.loc[data[col] > upper_bound, col] = upper_bound
                    self.logger.info(f"Capped {outlier_count} outliers in {col}")
        
        return data
    
    def _optimize_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types to reduce memory usage."""
        for col in data.columns:
            if data[col].dtype == 'object':
                # Convert to category if cardinality is low
                if data[col].nunique() / len(data) < 0.5:
                    data[col] = data[col].astype('category')
            elif data[col].dtype in ['int64', 'int32']:
                # Downcast integers
                data[col] = pd.to_numeric(data[col], downcast='integer')
            elif data[col].dtype in ['float64', 'float32']:
                # Downcast floats
                data[col] = pd.to_numeric(data[col], downcast='float')
        
        return data
    
    def create_insurance_features(self, data: pd.DataFrame = None) -> pd.DataFrame:
        """Create insurance-specific features."""
        if data is None:
            data = self.processed_data
        if data is None:
            raise ValueError("No data available for feature engineering.")
            
        feature_data = data.copy()
        
        # Loss Ratio
        if 'TotalClaims' in feature_data.columns and 'TotalPremium' in feature_data.columns:
            feature_data['LossRatio'] = feature_data['TotalClaims'] / (feature_data['TotalPremium'] + 1e-8)
            feature_data['ProfitMargin'] = feature_data['TotalPremium'] - feature_data['TotalClaims']
            feature_data['ClaimRate'] = (feature_data['TotalClaims'] > 0).astype(int)
        
        # Vehicle Age
        if 'RegistrationYear' in feature_data.columns:
            current_year = datetime.now().year
            feature_data['VehicleAge'] = current_year - feature_data['RegistrationYear']
            feature_data['VehicleAgeGroup'] = pd.cut(
                feature_data['VehicleAge'], 
                bins=[0, 3, 7, 15, 100], 
                labels=['New', 'Recent', 'Mature', 'Old']
            )
        
        # Premium per Sum Insured
        if 'TotalPremium' in feature_data.columns and 'SumInsured' in feature_data.columns:
            feature_data['PremiumRate'] = feature_data['TotalPremium'] / (feature_data['SumInsured'] + 1e-8)
        
        # Risk Score (composite)
        risk_factors = []
        if 'VehicleAge' in feature_data.columns:
            risk_factors.append('VehicleAge')
        if 'LossRatio' in feature_data.columns:
            risk_factors.append('LossRatio')
            
        if risk_factors:
            scaler = StandardScaler()
            risk_scores = scaler.fit_transform(feature_data[risk_factors].fillna(0))
            feature_data['RiskScore'] = np.mean(risk_scores, axis=1)
        
        self.logger.info(f"Created {len(feature_data.columns) - len(data.columns)} new features")
        return feature_data
    
    def encode_categorical_features(self, data: pd.DataFrame = None, 
                                  method: str = 'onehot') -> pd.DataFrame:
        """Encode categorical features."""
        if data is None:
            data = self.processed_data
        if data is None:
            raise ValueError("No data available for encoding.")
            
        encoded_data = data.copy()
        categorical_cols = encoded_data.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if method == 'onehot':
                # One-hot encoding for low cardinality
                if encoded_data[col].nunique() <= 10:
                    dummies = pd.get_dummies(encoded_data[col], prefix=col, drop_first=True)
                    encoded_data = pd.concat([encoded_data, dummies], axis=1)
                    encoded_data = encoded_data.drop(columns=[col])
                else:
                    # Label encoding for high cardinality
                    le = LabelEncoder()
                    encoded_data[col] = le.fit_transform(encoded_data[col])
                    self.encoders[col] = le
            
        return encoded_data
