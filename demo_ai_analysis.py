import pandas as pd
import numpy as np
from ai_analysis import SolarActivityPredictor, AIReportGenerator
from datetime import datetime, timedelta
import logging

def generate_sample_data(days=30):
    """Generate sample data for demonstration"""
    dates = pd.date_range(
        end=datetime.now(),
        periods=days * 24,
        freq='h'
    )
    
    # Generate synthetic solar activity data
    base_intensity = np.sin(np.linspace(0, 4*np.pi, len(dates)))
    noise = np.random.normal(0, 0.2, len(dates))
    trend = np.linspace(0, 0.5, len(dates))
    
    data = pd.DataFrame({
        'timestamp': dates,
        'intensity': base_intensity + noise + trend,
        'kp_index': np.random.uniform(0, 9, len(dates))
    })
    
    return data.set_index('timestamp')

def run_ai_demo():
    print("Starting AI Analysis Demo...")
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample data
    data = generate_sample_data()
    print("\nGenerated sample data for the last 30 days")
    logging.info(f"Data shape: {data.shape}")
    logging.info(f"Columns: {data.columns.tolist()}")
    
    # Initialize AI components
    predictor = SolarActivityPredictor()
    report_generator = AIReportGenerator()
    
    try:
        # Train the model
        print("\nTraining AI model...")
        print("Preparing data sequences...")
        training_metrics = predictor.train(data, epochs=10)
        print("\nModel Summary:")
        predictor.model.summary()
        print(f"Training completed - Loss: {training_metrics['loss']:.4f}")
        
        # Generate predictions and insights
        print("\nGenerating AI analysis report...")
        print("Making predictions for next 12 hours...")
        report = report_generator.generate_report(data)
        
        print("\n" + "="*60)
        print(report)
        print("="*60)
        
        if not report.startswith("Error"):
            print("\nDemo completed successfully!")
        else:
            print("\nDemo completed with errors. Check the report above.")
    except ValueError as e:
        logging.error(f"Error in model training/prediction: {e}")
        print("\nError: Model training failed. Check the logs for details.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        print("\nAn unexpected error occurred. Check the logs for details.")

if __name__ == "__main__":
    run_ai_demo() 