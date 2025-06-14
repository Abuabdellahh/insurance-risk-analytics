"""
EDA Utilities Module

Provides tools for exploratory data analysis, visualization, and statistical profiling.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class EDAAnalyzer:
    """
    EDA Analysis Tools for Insurance Data
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize EDA Analyzer
        
        Args:
            data: Input DataFrame for analysis
        """
        self.data = data
        self.numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        self.categorical_cols = self.data.select_dtypes(include=['object']).columns
        
    def generate_data_profile(self) -> Dict:
        """
        Generate comprehensive data profile
        
        Returns:
            Dictionary containing data statistics
        """
        profile = {
            'basic_info': {
                'rows': len(self.data),
                'columns': len(self.data.columns),
                'memory_usage': f"{self.data.memory_usage().sum() / 1024**2:.2f} MB"
            },
            'missing_values': {
                'total': self.data.isnull().sum().sum(),
                'percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict()
            },
            'numerical_stats': {},
            'categorical_stats': {}
        }
        
        # Add numerical statistics
        if len(self.numerical_cols) > 0:
            numerical_stats = self.data[self.numerical_cols].describe().to_dict()
            profile['numerical_stats'] = numerical_stats
        
        # Add categorical statistics
        if len(self.categorical_cols) > 0:
            for col in self.categorical_cols:
                profile['categorical_stats'][col] = {
                    'unique_values': self.data[col].nunique(),
                    'top_values': self.data[col].value_counts().head(5).to_dict()
                }
        
        return profile
    
    def plot_numerical_distribution(self, column: str, bins: int = 30) -> None:
        """
        Plot distribution of a numerical column
        
        Args:
            column: Column name to plot
            bins: Number of bins for histogram
        """
        if column not in self.numerical_cols:
            raise ValueError(f"Column {column} is not numerical")
            
        plt.figure(figsize=(12, 6))
        sns.histplot(self.data[column], bins=bins, kde=True)
        plt.title(f'Distribution of {column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    
    def plot_categorical_distribution(self, column: str) -> None:
        """
        Plot distribution of a categorical column
        
        Args:
            column: Column name to plot
        """
        if column not in self.categorical_cols:
            raise ValueError(f"Column {column} is not categorical")
            
        plt.figure(figsize=(12, 6))
        sns.countplot(data=self.data, x=column)
        plt.title(f'Distribution of {column}')
        plt.xticks(rotation=45)
        plt.show()
