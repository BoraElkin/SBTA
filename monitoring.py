import logging
import time
import psutil
import pandas as pd
from datetime import datetime
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class SystemMetrics:
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    timestamp: datetime

@dataclass
class ModelMetrics:
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    prediction_time: float
    timestamp: datetime

class MonitoringSystem:
    def __init__(self, config_path: str = "config.json"):
        self.config = self._load_config(config_path)
        self.metrics_dir = self.config.get("monitoring_path", "monitoring/")
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Initialize metrics storage
        self.system_metrics: Dict[str, SystemMetrics] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        
        # Set up logging
        logging.basicConfig(
            filename=os.path.join(self.metrics_dir, 'monitoring.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            logging.error(f"Error loading config: {e}")
            return {}

    def capture_system_metrics(self) -> SystemMetrics:
        """Capture current system performance metrics"""
        metrics = SystemMetrics(
            cpu_usage=psutil.cpu_percent(),
            memory_usage=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            timestamp=datetime.now()
        )
        
        self.system_metrics[metrics.timestamp.isoformat()] = metrics
        return metrics

    def record_model_metrics(self, 
                           accuracy: float,
                           precision: float,
                           recall: float,
                           f1_score: float,
                           prediction_time: float) -> ModelMetrics:
        """Record model performance metrics"""
        metrics = ModelMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            prediction_time=prediction_time,
            timestamp=datetime.now()
        )
        
        self.model_metrics[metrics.timestamp.isoformat()] = metrics
        return metrics

    def save_metrics(self):
        """Save all metrics to files"""
        # Save system metrics
        system_df = pd.DataFrame([
            {
                'timestamp': k,
                'cpu_usage': v.cpu_usage,
                'memory_usage': v.memory_usage,
                'disk_usage': v.disk_usage
            }
            for k, v in self.system_metrics.items()
        ])
        
        if not system_df.empty:
            system_df.to_csv(
                os.path.join(self.metrics_dir, 'system_metrics.csv'),
                index=False
            )

        # Save model metrics
        model_df = pd.DataFrame([
            {
                'timestamp': k,
                'accuracy': v.accuracy,
                'precision': v.precision,
                'recall': v.recall,
                'f1_score': v.f1_score,
                'prediction_time': v.prediction_time
            }
            for k, v in self.model_metrics.items()
        ])
        
        if not model_df.empty:
            model_df.to_csv(
                os.path.join(self.metrics_dir, 'model_metrics.csv'),
                index=False
            )

    def check_alerts(self) -> Optional[str]:
        """Check for any metric alerts"""
        alerts = []
        
        # Get latest system metrics
        if self.system_metrics:
            latest = list(self.system_metrics.values())[-1]
            
            if latest.cpu_usage > 90:
                alerts.append(f"HIGH CPU USAGE: {latest.cpu_usage}%")
            if latest.memory_usage > 90:
                alerts.append(f"HIGH MEMORY USAGE: {latest.memory_usage}%")
            if latest.disk_usage > 90:
                alerts.append(f"HIGH DISK USAGE: {latest.disk_usage}%")
        
        # Get latest model metrics
        if self.model_metrics:
            latest = list(self.model_metrics.values())[-1]
            
            if latest.accuracy < 0.7:
                alerts.append(f"LOW MODEL ACCURACY: {latest.accuracy}")
            if latest.prediction_time > 1.0:
                alerts.append(f"HIGH PREDICTION TIME: {latest.prediction_time}s")
        
        if alerts:
            alert_msg = "\n".join(alerts)
            logging.warning(f"System Alerts:\n{alert_msg}")
            return alert_msg
        
        return None

    def generate_report(self) -> str:
        """Generate a monitoring report"""
        report = ["Monitoring Report", "=" * 50, ""]
        
        # System metrics summary
        if self.system_metrics:
            metrics = list(self.system_metrics.values())
            avg_cpu = sum(m.cpu_usage for m in metrics) / len(metrics)
            avg_memory = sum(m.memory_usage for m in metrics) / len(metrics)
            
            report.extend([
                "System Metrics:",
                f"Average CPU Usage: {avg_cpu:.2f}%",
                f"Average Memory Usage: {avg_memory:.2f}%",
                ""
            ])
        
        # Model metrics summary
        if self.model_metrics:
            metrics = list(self.model_metrics.values())
            avg_accuracy = sum(m.accuracy for m in metrics) / len(metrics)
            avg_prediction_time = sum(m.prediction_time for m in metrics) / len(metrics)
            
            report.extend([
                "Model Metrics:",
                f"Average Accuracy: {avg_accuracy:.2f}",
                f"Average Prediction Time: {avg_prediction_time:.3f}s",
                ""
            ])
        
        report_text = "\n".join(report)
        
        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.metrics_dir, f'report_{timestamp}.txt')
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        return report_text 