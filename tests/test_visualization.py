import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from visualization import (
    plot_solar_intensity,
    plot_kp_index,
    plot_event_distribution,
    plot_location_heatmap,
    visualize_data
)
from io import StringIO
import logging

class TestVisualization(unittest.TestCase):
    def setUp(self):
        """Set up test data and capture logging output"""
        # Create sample data
        dates = pd.date_range(start='2024-01-01', periods=100, freq='H')
        self.test_data = pd.DataFrame({
            'timestamp': dates,
            'intensity': np.random.exponential(2, 100),
            'kp_index': np.random.uniform(0, 9, 100),
            'event': np.random.choice([0, 1], 100),
            'location': np.random.choice(['N30E45', 'S15W30', 'N00W00'], 100)
        })

        # Capture logging output
        self.log_output = StringIO()
        self.log_handler = logging.StreamHandler(self.log_output)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.INFO)

    def tearDown(self):
        """Clean up after each test"""
        plt.close('all')  # Close all figures
        logging.getLogger().removeHandler(self.log_handler)
        self.log_output.close()

    def test_plot_solar_intensity(self):
        """Test solar intensity plotting"""
        fig, ax = plt.subplots()
        plot_solar_intensity(ax, self.test_data)
        
        # Check if plot contains data
        self.assertTrue(len(ax.lines) > 0)
        self.assertEqual(ax.get_xlabel(), "Time")
        self.assertEqual(ax.get_ylabel(), "Intensity")
        
        # Test with missing column
        bad_data = self.test_data.drop(columns=['intensity'])
        plot_solar_intensity(ax, bad_data)
        self.assertIn("Column 'intensity' not found", self.log_output.getvalue())

    def test_plot_kp_index(self):
        """Test Kp index plotting"""
        fig, ax = plt.subplots()
        plot_kp_index(ax, self.test_data)
        
        self.assertTrue(len(ax.lines) > 0)
        self.assertEqual(ax.get_xlabel(), "Time")
        self.assertEqual(ax.get_ylabel(), "Kp Index")

    def test_plot_event_distribution(self):
        """Test event distribution plotting"""
        fig, ax = plt.subplots()
        plot_event_distribution(ax, self.test_data)
        
        # Check if bars are created
        self.assertTrue(len(ax.patches) == 2)  # Two bars for binary events
        self.assertEqual(ax.get_ylabel(), "Count")

    def test_plot_location_heatmap(self):
        """Test location heatmap plotting"""
        fig, ax = plt.subplots()
        plot_location_heatmap(ax, self.test_data)
        
        # Check if heatmap is created
        self.assertTrue(len(ax.collections) > 0)
        self.assertEqual(ax.get_xlabel(), "East-West Position")
        self.assertEqual(ax.get_ylabel(), "North-South Position")

    def test_visualize_data_complete(self):
        """Test complete visualization function"""
        # Test with valid data
        visualize_data(self.test_data)
        self.assertIn("Generated comprehensive solar activity visualization", 
                     self.log_output.getvalue())
        
        # Check if file was saved
        import os
        self.assertTrue(os.path.exists('solar_activity_dashboard.png'))
        os.remove('solar_activity_dashboard.png')  # Clean up

    def test_visualize_data_empty(self):
        """Test visualization with empty data"""
        empty_df = pd.DataFrame()
        visualize_data(empty_df)
        self.assertIn("Input DataFrame is empty", self.log_output.getvalue())

    def test_visualize_data_missing_columns(self):
        """Test visualization with missing required columns"""
        incomplete_data = self.test_data.drop(columns=['timestamp'])
        visualize_data(incomplete_data)
        self.assertIn("Required column 'timestamp' not found", 
                     self.log_output.getvalue())

if __name__ == '__main__':
    unittest.main() 