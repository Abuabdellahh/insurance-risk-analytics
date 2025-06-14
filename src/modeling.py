"""
Modeling Module

Provides tools for building and evaluating predictive models for insurance risk analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
import lightgbm as lgb
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class InsuranceModeler:
    """
    Insurance Risk Modeling Tools
    """
    
    def __init__(self, data: pd.DataFrame, target: str):
        """
        Initialize Modeler
        
        Args:
            data: Input DataFrame
            target: Target variable name
        """
        self.data = data
        self.target = target
        self.X = self.data.drop(columns=[target])
        self.y = self.data[target]
        self.numerical_features = self.X.select_dtypes(include=[np.number]).columns
        self.categorical_features = self.X.select_dtypes(include=['object']).columns
        self.models = {}
        
    def prepare_data(self, test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for modeling
        
        Args:
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        # Create preprocessing pipeline
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numerical_features),
                ('cat', OneHotEncoder(), self.categorical_features)
            ]
        )
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )
        
        # Apply preprocessing
        X_train = preprocessor.fit_transform(X_train)
        X_test = preprocessor.transform(X_test)
        
        return X_train, X_test, y_train, y_test, preprocessor
    
    def train_linear_regression(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """
        Train Linear Regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Dictionary containing model and evaluation metrics
        """
        from sklearn.linear_model import LinearRegression
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        self.models['linear_regression'] = model
        
        return {
            'model': model,
            'r2': model.score(X_train, y_train),
            'coefficients': model.coef_,
            'intercept': model.intercept_
        }
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     params: Dict = None) -> Dict:
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training target
            params: Model parameters
            
        Returns:
            Dictionary containing model and evaluation metrics
        """
        if params is None:
            params = {
                'objective': 'reg:squarederror',
                'n_estimators': 100,
                'max_depth': 6,
                'learning_rate': 0.1
            }
        
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        
        self.models['xgboost'] = model
        
        return {
            'model': model,
            'feature_importances': model.feature_importances_,
            'best_params': model.get_params()
        }
    
    def evaluate_model(self, model: object, X_test: np.ndarray, 
                      y_test: np.ndarray) -> Dict:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary containing evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        return {
            'r2': r2_score(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred)
        }
    
    def compare_models(self, X_test: np.ndarray, y_test: np.ndarray) -> pd.DataFrame:
        """
        Compare performance of all trained models
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            DataFrame with model comparison results
        """
        results = []
        for model_name, model in self.models.items():
            metrics = self.evaluate_model(model, X_test, y_test)
            results.append({
                'model': model_name,
                **metrics
            })
            
        return pd.DataFrame(results)
