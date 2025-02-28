import time
from monitoring import MonitoringSystem
import numpy as np
from datetime import datetime
import logging

def simulate_model_prediction():
    """Simulate model prediction with varying performance"""
    # Simulate some variation in model performance
    accuracy = np.random.normal(0.85, 0.1)  # Mean 0.85, std 0.1
    precision = np.random.normal(0.83, 0.1)
    recall = np.random.normal(0.87, 0.1)
    f1_score = np.random.normal(0.85, 0.1)
    
    # Simulate prediction time with occasional spikes
    prediction_time = np.random.exponential(0.3)
    
    return {
        'accuracy': max(min(accuracy, 1.0), 0.0),  # Clip between 0 and 1
        'precision': max(min(precision, 1.0), 0.0),
        'recall': max(min(recall, 1.0), 0.0),
        'f1_score': max(min(f1_score, 1.0), 0.0),
        'prediction_time': prediction_time
    }

def run_monitoring_demo(duration_seconds=60, interval_seconds=5):
    """Run a demonstration of the monitoring system"""
    print(f"Starting monitoring demo for {duration_seconds} seconds...")
    
    # Initialize monitoring system
    monitoring = MonitoringSystem()
    
    start_time = time.time()
    iteration = 0
    
    try:
        while time.time() - start_time < duration_seconds:
            iteration += 1
            print(f"\nIteration {iteration}")
            print("-" * 20)
            
            # Capture system metrics
            system_metrics = monitoring.capture_system_metrics()
            print(f"CPU Usage: {system_metrics.cpu_usage:.1f}%")
            print(f"Memory Usage: {system_metrics.memory_usage:.1f}%")
            print(f"Disk Usage: {system_metrics.disk_usage:.1f}%")
            
            # Simulate and record model metrics
            model_perf = simulate_model_prediction()
            monitoring.record_model_metrics(**model_perf)
            print(f"Model Accuracy: {model_perf['accuracy']:.3f}")
            print(f"Prediction Time: {model_perf['prediction_time']:.3f}s")
            
            # Check for alerts
            alerts = monitoring.check_alerts()
            if alerts:
                print("\n⚠️ ALERTS:")
                print(alerts)
            
            # Save metrics every 5 iterations
            if iteration % 5 == 0:
                monitoring.save_metrics()
                print("\nMetrics saved to CSV files")
                
                # Generate and print report
                print("\n📊 MONITORING REPORT:")
                print(monitoring.generate_report())
            
            time.sleep(interval_seconds)
    
    except KeyboardInterrupt:
        print("\nMonitoring demo interrupted by user")
    finally:
        # Save final metrics and generate report
        monitoring.save_metrics()
        final_report = monitoring.generate_report()
        print("\n📊 FINAL MONITORING REPORT:")
        print(final_report)
        
        print("\nDemo completed. Check the 'monitoring' directory for detailed logs and metrics.")

if __name__ == "__main__":
    # Run the demo for 60 seconds with 5-second intervals
    run_monitoring_demo(60, 5) 