import unittest
import os
import shutil
import json
from datetime import datetime
from monitoring import MonitoringSystem, SystemMetrics, ModelMetrics

class TestMonitoring(unittest.TestCase):
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "monitoring_path": "test_monitoring/",
            "monitoring": {
                "enabled": True,
                "interval": 300
            }
        }
        
        # Create test config file
        with open("test_config.json", "w") as f:
            json.dump(self.test_config, f)
            
        self.monitoring = MonitoringSystem("test_config.json")

    def tearDown(self):
        """Clean up test environment"""
        if os.path.exists("test_monitoring"):
            shutil.rmtree("test_monitoring")
        if os.path.exists("test_config.json"):
            os.remove("test_config.json")

    def test_capture_system_metrics(self):
        """Test system metrics capture"""
        metrics = self.monitoring.capture_system_metrics()
        self.assertIsInstance(metrics, SystemMetrics)
        self.assertTrue(0 <= metrics.cpu_usage <= 100)
        self.assertTrue(0 <= metrics.memory_usage <= 100)
        self.assertTrue(0 <= metrics.disk_usage <= 100)

    def test_record_model_metrics(self):
        """Test model metrics recording"""
        metrics = self.monitoring.record_model_metrics(
            accuracy=0.85,
            precision=0.83,
            recall=0.87,
            f1_score=0.85,
            prediction_time=0.15
        )
        self.assertIsInstance(metrics, ModelMetrics)
        self.assertEqual(metrics.accuracy, 0.85)
        self.assertEqual(metrics.prediction_time, 0.15)

    def test_save_metrics(self):
        """Test metrics saving"""
        # Add some test metrics
        self.monitoring.capture_system_metrics()
        self.monitoring.record_model_metrics(0.85, 0.83, 0.87, 0.85, 0.15)
        
        # Save metrics
        self.monitoring.save_metrics()
        
        # Check if files were created
        self.assertTrue(os.path.exists(os.path.join(
            self.test_config["monitoring_path"], 
            "system_metrics.csv"
        )))
        self.assertTrue(os.path.exists(os.path.join(
            self.test_config["monitoring_path"], 
            "model_metrics.csv"
        )))

    def test_check_alerts(self):
        """Test alert checking"""
        # Record metrics that should trigger alerts
        self.monitoring.system_metrics[datetime.now().isoformat()] = SystemMetrics(
            cpu_usage=95,
            memory_usage=95,
            disk_usage=95,
            timestamp=datetime.now()
        )
        
        alerts = self.monitoring.check_alerts()
        self.assertIsNotNone(alerts)
        self.assertIn("HIGH CPU USAGE", alerts)
        self.assertIn("HIGH MEMORY USAGE", alerts)

    def test_generate_report(self):
        """Test report generation"""
        # Add some test metrics
        self.monitoring.capture_system_metrics()
        self.monitoring.record_model_metrics(0.85, 0.83, 0.87, 0.85, 0.15)
        
        report = self.monitoring.generate_report()
        self.assertIn("System Metrics:", report)
        self.assertIn("Model Metrics:", report)

if __name__ == '__main__':
    unittest.main() 