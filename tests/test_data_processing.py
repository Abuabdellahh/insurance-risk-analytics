import unittest
import pandas as pd
import numpy as np
from src.data_processing import DataProcessor

class TestDataProcessor(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.sample_data = pd.DataFrame({
            'age': [25, 30, 35, 40, 45],
            'gender': ['M', 'F', 'M', 'F', 'M'],
            'claim_amount': [1000, 2000, 3000, 4000, 5000],
            'vehicle_age': [2, 3, 5, 1, 4]
        })
        self.processor = DataProcessor()

    def test_load_data(self):
        # Test data loading
        df = self.processor.load_data('test_data.csv')
        self.assertIsNotNone(df)
        self.assertIsInstance(df, pd.DataFrame)

    def test_clean_data(self):
        # Test data cleaning
        cleaned_df = self.processor.clean_data(
            remove_duplicates=True,
            handle_missing='auto',
            outlier_method='iqr'
        )
        self.assertIsNotNone(cleaned_df)
        self.assertIsInstance(cleaned_df, pd.DataFrame)

    def test_create_insurance_features(self):
        # Test feature engineering
        feature_df = self.processor.create_insurance_features()
        self.assertIsNotNone(feature_df)
        self.assertIn('loss_ratio', feature_df.columns)
        self.assertIn('vehicle_age_group', feature_df.columns)

    def test_encode_categorical_features(self):
        # Test categorical encoding
        encoded_df = self.processor.encode_categorical_features(method='onehot')
        self.assertIsNotNone(encoded_df)
        self.assertIn('gender_F', encoded_df.columns)
        self.assertIn('gender_M', encoded_df.columns)

if __name__ == '__main__':
    unittest.main()
