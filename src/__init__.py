"""
Insurance Risk Analytics Package

A comprehensive toolkit for insurance risk analysis, hypothesis testing,
and predictive modeling.
"""

__version__ = "1.0.0"
__author__ = "Insurance Analytics Team"
__email__ = "analytics@alphacare.com"

from .data_processing import DataProcessor
from .eda_utils import EDAAnalyzer
from .statistical_tests import HypothesisTestSuite
from .modeling import InsuranceModeler
from .visualization import InsuranceVisualizer

__all__ = [
    "DataProcessor",
    "EDAAnalyzer",
    "HypothesisTestSuite",
    "InsuranceModeler",
    "InsuranceVisualizer"
]
