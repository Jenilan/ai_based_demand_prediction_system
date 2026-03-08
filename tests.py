from django.test import TestCase
import pandas as pd
import numpy as np
import tempfile
import os
from .predictor import DemandPredictor

class PredictorTests(TestCase):
    def setUp(self):
        # Create a sample CSV
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.csv_path = os.path.join(self.tmp_dir.name, 'sales.csv')
        
        # Create 20 months of data
        dates = pd.date_range(start='2020-01-31', periods=20, freq='ME')
        # Ensure positive sales to avoid dropping rows
        sales = np.arange(20) * 10 + 100 + np.random.normal(0, 5, 20) 
        df = pd.DataFrame({'date': dates, 'sales': sales})
        df.to_csv(self.csv_path, index=False)
        
        self.predictor = DemandPredictor(n_lags=3)

    def tearDown(self):
        self.tmp_dir.cleanup()

    def test_load_and_validate(self):
        df, _ = self.predictor._load_and_validate_data(self.csv_path, date_col='date', sales_col='sales', price_col=None)
        self.assertEqual(len(df), 20)
        self.assertIn('date', df.columns)
        self.assertIn('sales', df.columns)

    def test_end_to_end_analysis(self):
        result = self.predictor.analyze(self.csv_path)
        
        self.assertIn('predicted_demand', result)
        self.assertIn('suggested_stock', result)
        self.assertTrue(result['predicted_demand'] > 0)
        
        # Check history
        self.assertIn('history', result)
        self.assertEqual(len(result['history']['sales']), 20)

    def test_insufficient_data(self):
        # Create small CSV
        small_csv = os.path.join(self.tmp_dir.name, 'small.csv')
        dates = pd.date_range(start='2020-01-31', periods=3, freq='ME')
        sales = [10, 20, 30]
        pd.DataFrame({'date': dates, 'sales': sales}).to_csv(small_csv, index=False)
        
        result = self.predictor.analyze(small_csv)
        self.assertIn('average', result['model_info']['model_used'].lower())

    def test_item_level_analysis(self):
        # Create CSV with item column
        item_csv = os.path.join(self.tmp_dir.name, 'item_sales.csv')
        dates = pd.date_range(start='2020-01-31', periods=12, freq='ME')
        data = []
        for date in dates:
            data.extend([
                {'date': date, 'sales': 10, 'item': 'A'},
                {'date': date, 'sales': 15, 'item': 'B'},
            ])
        df = pd.DataFrame(data)
        df.to_csv(item_csv, index=False)
        
        result = self.predictor.analyze(item_csv)
        self.assertIn('top_products', result)
        self.assertIn('restock_recs', result)

    def test_column_inference(self):
        # Test automatic column detection
        infer_csv = os.path.join(self.tmp_dir.name, 'infer.csv')
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2020-01-01', periods=10, freq='ME'),
            'quantity': np.arange(10) + 10,
            'revenue': (np.arange(10) + 10) * 5
        })
        df.to_csv(infer_csv, index=False)
        
        df_loaded, _ = self.predictor._load_and_validate_data(infer_csv, None, None, None, None)
        self.assertIn('date', df_loaded.columns)
        self.assertIn('sales', df_loaded.columns)

    def test_error_handling(self):
        # Test with invalid CSV
        invalid_csv = os.path.join(self.tmp_dir.name, 'invalid.csv')
        with open(invalid_csv, 'w') as f:
            f.write('not,csv,data\n')
        
        with self.assertRaises(ValueError):
            self.predictor.analyze(invalid_csv)

    def test_null_value_cleaning(self):
        # Create data with missing and formatted numbers
        bad_csv = os.path.join(self.tmp_dir.name, 'bad.csv')
        df = pd.DataFrame({
            'Date': pd.date_range('2021-01-01', periods=6, freq='ME'),
            'Qty Sold': ['10', '20', '', '30', '40', '50'],
            'Price($)': ['$1,000', '2,000', '3000', '', '$4,000', '5000']
        })
        df.to_csv(bad_csv, index=False)

        cleaned_df, _ = self.predictor._load_and_validate_data(bad_csv, None, None, None, None)
        # ensure no NaNs in sales/price
        self.assertFalse(cleaned_df['sales'].isna().any())
        self.assertFalse(cleaned_df['price'].isna().any())
        # blank values should have been set to 0
        self.assertEqual(cleaned_df['sales'].iloc[2], 0)
        self.assertEqual(cleaned_df['price'].iloc[3], 0)

        # running analysis should not crash
        result = self.predictor.analyze(bad_csv)
        self.assertIsInstance(result.get('predicted_demand'), float)

    def test_varied_format_inference(self):
        infer_csv = os.path.join(self.tmp_dir.name, 'infer2.csv')
        df = pd.DataFrame({
            'Sale-Date': pd.date_range('2022-01-01', periods=5, freq='ME'),
            'Units': ['1,000', '1.5k', '2000', '2500', '3,000'],
            'Revenue': ['1000', '1500', '2000', '2500', '3000']
        })
        df.to_csv(infer_csv, index=False)
        cleaned_df, _ = self.predictor._load_and_validate_data(infer_csv, None, None, None, None)
        # The inference should pick up columns correctly
        self.assertIn('date', cleaned_df.columns)
        self.assertIn('sales', cleaned_df.columns)
        # numeric cleaning should convert '1.5k' -> 1.5 (coerced to n/a then zero) so at least not NaN
        self.assertFalse(cleaned_df['sales'].isna().any())

    def test_partial_bad_dates(self):
        bad_dates_csv = os.path.join(self.tmp_dir.name, 'bad_dates.csv')
        df = pd.DataFrame({
            'date': ['2021-01-31', 'not a date', '2021-03-31'],
            'sales': [10, 20, 30]
        })
        df.to_csv(bad_dates_csv, index=False)
        cleaned_df, _ = self.predictor._load_and_validate_data(bad_dates_csv, 'date', 'sales', None, None)
        # row with invalid date should be removed
        self.assertEqual(len(cleaned_df), 2)
        self.assertNotIn('not a date', cleaned_df['date'].astype(str).tolist())

    def test_cache_reuse(self):
        # ensure calling analyze twice on same file returns same cached object
        cache_csv = os.path.join(self.tmp_dir.name, 'cache.csv')
        dates = pd.date_range(start='2020-01-31', periods=12, freq='ME')
        sales = np.arange(12) * 5 + 50
        pd.DataFrame({'date': dates, 'sales': sales}).to_csv(cache_csv, index=False)
        result1 = self.predictor.analyze(cache_csv)
        result2 = self.predictor.analyze(cache_csv)
        self.assertIs(result1, result2)
