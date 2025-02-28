import unittest
import os
import json
import shutil
import pandas as pd
from unittest.mock import patch, MagicMock
from main import (
    load_config,
    setup_directories,
    generate_report,
    validate_data,
    main,
    ConfigurationError
)

class TestMainOrchestration(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "use_sample_data": True,
            "save_visualizations": True,
            "visualization_path": "test_visualizations/",
            "report_path": "test_reports/",
            "model_path": "test_models/",
            "data_cache_path": "test_cache/"
        }
        
        # Create a temporary config file
        with open("test_config.json", "w") as f:
            json.dump(self.test_config, f)

    def tearDown(self):
        """Clean up test environment"""
        # Remove test directories and files
        for path in [
            "test_visualizations",
            "test_reports",
            "test_models",
            "test_cache"
        ]:
            if os.path.exists(path):
                shutil.rmtree(path)
        
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")

    def test_load_config(self):
        """Test configuration loading"""
        # Test loading existing config
        config = load_config("test_config.json")
        self.assertEqual(config, self.test_config)
        
        # Test creating default config
        os.remove("test_config.json")
        config = load_config("test_config.json")
        self.assertTrue(all(key in config for key in self.test_config.keys()))

    def test_setup_directories(self):
        """Test directory creation"""
        setup_directories(self.test_config)
        for path in self.test_config.values():
            if isinstance(path, str) and path.endswith('/'):
                self.assertTrue(os.path.exists(path))

    def test_generate_report(self):
        """Test report generation"""
        setup_directories(self.test_config)
        report_file = generate_report(
            "Test report content",
            self.test_config,
            "test"
        )
        
        self.assertTrue(os.path.exists(report_file))
        with open(report_file, 'r') as f:
            content = f.read()
            self.assertIn("Test report content", content)
            self.assertIn("Space Weather Analysis Report", content)

    def test_validate_data(self):
        """Test data validation"""
        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'intensity': [1.0],
            'location': ['N30E45']
        })
        self.assertTrue(validate_data(valid_data))
        
        # Missing column
        invalid_data = valid_data.drop(columns=['intensity'])
        with self.assertRaises(ValueError):
            validate_data(invalid_data)
        
        # Invalid timestamp type
        invalid_data = valid_data.copy()
        invalid_data['timestamp'] = ['2024-01-01']
        with self.assertRaises(ValueError):
            validate_data(invalid_data)

    @patch('main.fetch_solar_data')
    @patch('main.preprocess_data')
    @patch('main.train_model')
    @patch('main.predict_event')
    @patch('main.visualize_data')
    def test_main_execution(self, mock_viz, mock_predict, mock_train, 
                          mock_preprocess, mock_fetch):
        """Test main execution flow"""
        # Mock return values
        mock_fetch.return_value = pd.DataFrame({
            'timestamp': [pd.Timestamp.now()],
            'intensity': [1.0],
            'location': ['N30E45']
        })
        mock_preprocess.return_value = pd.DataFrame({'processed': [1]})
        mock_train.return_value = MagicMock()
        mock_predict.return_value = [1]
        
        # Run main function
        result = main()
        
        # Verify execution
        self.assertTrue(result)
        mock_fetch.assert_called_once()
        mock_preprocess.assert_called_once()
        mock_train.assert_called_once()
        mock_predict.assert_called_once()
        mock_viz.assert_called_once()

    def test_main_error_handling(self):
        """Test main function error handling"""
        # Test with empty data
        with patch('main.fetch_solar_data', return_value=pd.DataFrame()):
            result = main()
            self.assertFalse(result)
        
        # Test with invalid data
        with patch('main.fetch_solar_data', return_value=pd.DataFrame({'invalid': [1]})):
            result = main()
            self.assertFalse(result)

if __name__ == '__main__':
    unittest.main() 