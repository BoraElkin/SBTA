import unittest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import patch, MagicMock
from data_fetch import (
    load_api_keys,
    fetch_nasa_data,
    fetch_noaa_data,
    generate_sample_data,
    fetch_solar_data,
    parse_time
)
import warnings
import logging
from io import StringIO

class TestDataFetch(unittest.TestCase):
    def setUp(self):
        """Set up any necessary test fixtures"""
        # Capture logging output
        self.log_output = StringIO()
        self.log_handler = logging.StreamHandler(self.log_output)
        logging.getLogger().addHandler(self.log_handler)
        logging.getLogger().setLevel(logging.WARNING)
        
        self.mock_nasa_response = [
            {
                "flrID": "2024-01-01T00:00:00-FLR-001",
                "beginTime": "2024-01-01T00:00:00",
                "endTime": "2024-01-01T00:10:00",
                "classType": "M5.2",
                "sourceLocation": "N30E45"
            },
            {
                "flrID": "2024-01-01T12:00:00-FLR-002",
                "beginTime": "2024-01-01T12:00:00",
                "endTime": "2024-01-01T12:15:00",
                "classType": "X1.1",
                "sourceLocation": "S15W30"
            }
        ]
        
        self.mock_noaa_content = """
# Solar-Geophysical Activity Report
2024 01 01
00:00    3.8     5.2
06:00:00    4.2     5.5
12:00    4.5     6.1
18:00:00    3.9     5.8
"""

    def tearDown(self):
        """Clean up after each test"""
        logging.getLogger().removeHandler(self.log_handler)
        self.log_output.close()

    def test_generate_sample_data(self):
        """Test sample data generation"""
        n_samples = 50
        data = generate_sample_data(n_samples)
        
        # Check basic properties
        self.assertEqual(len(data), n_samples)
        self.assertTrue(all(col in data.columns 
                          for col in ['timestamp', 'intensity', 'kp_index', 'location', 'event']))
        
        # Check data types
        self.assertTrue(isinstance(data['timestamp'][0], pd.Timestamp))
        self.assertTrue(isinstance(data['intensity'][0], float))
        self.assertTrue(isinstance(data['kp_index'][0], float))
        self.assertTrue(isinstance(data['location'][0], str))
        self.assertTrue(isinstance(data['event'][0], (int, np.int32, np.int64)))  # Allow numpy integer types

    @patch('requests.get')
    def test_fetch_nasa_data(self, mock_get):
        """Test NASA data fetching"""
        # Mock the API response
        mock_response = MagicMock()
        mock_response.json.return_value = self.mock_nasa_response
        mock_get.return_value = mock_response
        
        nasa_df = fetch_nasa_data()
        
        # Verify the data
        self.assertEqual(len(nasa_df), 2)
        self.assertTrue(all(col in nasa_df.columns 
                          for col in ['timestamp', 'intensity', 'duration', 'location']))
        
        # Check intensity conversion
        self.assertEqual(nasa_df['intensity'].iloc[0], 5.2)  # M5.2
        self.assertEqual(nasa_df['intensity'].iloc[1], 11.0)  # X1.1 (multiplied by 10)

    @patch('boto3.client')
    def test_fetch_noaa_data(self, mock_boto3_client):
        """Test NOAA data fetching"""
        # Clear previous log output
        self.log_output.seek(0)
        self.log_output.truncate()
        
        # Mock S3 client and responses
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3
        
        # Mock list_objects_v2
        mock_s3.list_objects_v2.return_value = {
            'Contents': [{'Key': 'test_key', 'LastModified': datetime.now()}]
        }
        
        # Mock get_object
        mock_s3.get_object.return_value = {
            'Body': MagicMock(read=lambda: self.mock_noaa_content.encode())
        }
        
        noaa_df = fetch_noaa_data()
        
        # Verify no unexpected warnings
        log_content = self.log_output.getvalue()
        self.assertNotIn("error", log_content.lower())
        self.assertNotIn("warning", log_content.lower())
        
        # Verify the data
        self.assertEqual(len(noaa_df), 4)  # Should handle both time formats
        self.assertTrue(all(col in noaa_df.columns 
                          for col in ['timestamp', 'kp_index', 'intensity']))
        
        # Verify first row values
        first_row = noaa_df.iloc[0]
        self.assertEqual(first_row['intensity'], 3.8)
        self.assertEqual(first_row['kp_index'], 5.2)
        
        # Verify timestamps are parsed correctly
        expected_times = [
            "2024-01-01 00:00:00",
            "2024-01-01 06:00:00",
            "2024-01-01 12:00:00",
            "2024-01-01 18:00:00"
        ]
        actual_times = noaa_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S').tolist()
        self.assertEqual(actual_times, expected_times)

    def test_fetch_solar_data(self):
        """Test combined data fetching"""
        # Test with sample data
        data = fetch_solar_data(use_sample_data=True)
        self.assertFalse(data.empty)
        self.assertTrue(all(col in data.columns 
                          for col in ['timestamp', 'intensity', 'kp_index', 'location']))

    def test_parse_time(self):
        """Test time string parsing with different formats"""
        base_date = pd.to_datetime('2024-01-01')
        
        # Test valid formats
        test_cases = [
            ('14:30', '2024-01-01 14:30:00'),
            ('14:30:45', '2024-01-01 14:30:45'),
            ('00:00', '2024-01-01 00:00:00'),
            ('23:59:59', '2024-01-01 23:59:59')
        ]
        
        for input_time, expected_output in test_cases:
            with self.subTest(input_time=input_time):
                result = parse_time(input_time, base_date)
                self.assertEqual(
                    result.strftime('%Y-%m-%d %H:%M:%S'),
                    expected_output
                )
        
        # Test invalid formats with warning capture
        invalid_formats = {
            '14.30': "Invalid time format '14.30': missing colon separator",
            '25:00': "Could not parse time string '25:00' in any known format",
            '14:60': "Could not parse time string '14:60' in any known format",
            'invalid': "Invalid time format 'invalid': missing colon separator"
        }
        
        for invalid_time, expected_message in invalid_formats.items():
            with self.subTest(invalid_time=invalid_time):
                # Clear previous log output
                self.log_output.seek(0)
                self.log_output.truncate()
                
                result = parse_time(invalid_time, base_date)
                self.assertIsNone(result)
                self.assertIn(
                    expected_message,
                    self.log_output.getvalue()
                )

if __name__ == '__main__':
    unittest.main() 