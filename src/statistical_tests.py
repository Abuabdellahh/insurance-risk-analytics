"""
Statistical Tests Module

Provides tools for hypothesis testing and statistical analysis in insurance risk analytics.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings('ignore')

class HypothesisTestSuite:
    """
    Statistical Hypothesis Testing Tools for Insurance Risk Analysis
    """
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize Hypothesis Test Suite
        
        Args:
            data: Input DataFrame for analysis
        """
        self.data = data
        
    def chi_square_test(self, var1: str, var2: str) -> Dict:
        """
        Perform Chi-Square test for independence between two categorical variables
        
        Args:
            var1: First categorical variable
            var2: Second categorical variable
            
        Returns:
            Dictionary containing test results
        """
        contingency_table = pd.crosstab(self.data[var1], self.data[var2])
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        
        return {
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected,
            'contingency_table': contingency_table
        }
    
    def t_test(self, group1: pd.Series, group2: pd.Series) -> Dict:
        """
        Perform independent samples t-test
        
        Args:
            group1: First sample
            group2: Second sample
            
        Returns:
            Dictionary containing test results
        """
        result = stats.ttest_ind(group1, group2, equal_var=False)
        
        return {
            't_statistic': result.statistic,
            'p_value': result.pvalue,
            'mean_diff': group1.mean() - group2.mean(),
            'cohen_d': self._calculate_cohen_d(group1, group2)
        }
    
    def anova_test(self, dependent_var: str, independent_var: str) -> Dict:
        """
        Perform ANOVA test for multiple groups
        
        Args:
            dependent_var: Continuous dependent variable
            independent_var: Categorical independent variable
            
        Returns:
            Dictionary containing test results
        """
        groups = []
        for category in self.data[independent_var].unique():
            groups.append(self.data[self.data[independent_var] == category][dependent_var])
            
        f_statistic, p_value = stats.f_oneway(*groups)
        
        return {
            'f_statistic': f_statistic,
            'p_value': p_value,
            'groups': len(groups)
        }
    
    def correlation_analysis(self, var1: str, var2: str, method: str = 'pearson') -> Dict:
        """
        Perform correlation analysis between two variables
        
        Args:
            var1: First variable
            var2: Second variable
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            Dictionary containing correlation results
        """
        if method not in ['pearson', 'spearman', 'kendall']:
            raise ValueError(f"Invalid method: {method}")
            
        corr_func = getattr(stats, f"{method}_r")
        corr, p_value = corr_func(self.data[var1], self.data[var2])
        
        return {
            'correlation': corr,
            'p_value': p_value,
            'method': method
        }
    
    def _calculate_cohen_d(self, group1: pd.Series, group2: pd.Series) -> float:
        """
        Calculate Cohen's d effect size
        """
        diff = group1.mean() - group2.mean()
        n1, n2 = len(group1), len(group2)
        var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
        pooled_var = ((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2)
        d = diff / np.sqrt(pooled_var)
        return d
