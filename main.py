import logging
import pandas as pd
import json
import os
from datetime import datetime
from data_fetch import fetch_solar_data
from preprocess import preprocess_data
from train_model import train_model
from predict import predict_event
from visualization import visualize_data
import matplotlib.pyplot as plt

class ConfigurationError(Exception):
    """Custom exception for configuration errors"""
    pass

def load_config(config_file="config.json"):
    """Load configuration from JSON file"""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        # Create default configuration
        default_config = {
            "use_sample_data": True,
            "save_visualizations": True,
            "visualization_path": "visualizations/",
            "report_path": "reports/",
            "model_path": "models/",
            "data_cache_path": "cache/"
        }
        # Save default configuration
        config_dir = os.path.dirname(config_file)
        if config_dir:  # Only create directory if path contains directory
            os.makedirs(config_dir, exist_ok=True)
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=4)
        return default_config

def setup_directories(config):
    """Create necessary directories"""
    for path in [config["visualization_path"], 
                config["report_path"], 
                config["model_path"], 
                config["data_cache_path"]]:
        os.makedirs(path, exist_ok=True)

def generate_report(report_text, config, report_type="general"):
    """Generate timestamped reports"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(
        config["report_path"], 
        f"{report_type}_report_{timestamp}.txt"
    )
    
    with open(report_file, "w") as file:
        file.write(f"Space Weather Analysis Report\n")
        file.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write("-" * 50 + "\n\n")
        file.write(report_text + "\n")
    
    logging.info("Report generated: %s", report_file)
    return report_file

def validate_data(data):
    """Validate the required columns and data types"""
    required_columns = ['timestamp', 'intensity', 'location']
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    if not isinstance(data['timestamp'].iloc[0], pd.Timestamp):
        raise ValueError("'timestamp' column must contain datetime values")
    
    return True

def main():
    try:
        # Load configuration
        config = load_config()
        
        # Set up directories
        setup_directories(config)
        
        # Set up logging
        log_file = os.path.join(config["report_path"], "application.log")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        logging.info("Starting AI-Powered Space Weather Prediction MVP")
        
        # Step 1: Data Collection
        logging.info("Fetching solar activity data...")
        data = fetch_solar_data(use_sample_data=config["use_sample_data"])
        
        if data.empty:
            raise ValueError("Fetched data is empty")
        
        # Validate data structure
        validate_data(data)
        
        logging.info("Data fetched successfully with %d records", len(data))
        generate_report(
            f"Data Collection: Fetched {len(data)} records of solar activity data",
            config,
            "data_collection"
        )

        # Step 2: Data Preprocessing
        logging.info("Preprocessing data...")
        processed_data = preprocess_data(data)
        if processed_data.empty:
            raise ValueError("Data preprocessing resulted in empty dataset")
        
        generate_report(
            "Data Preprocessing: Completed data cleaning and normalization\n"
            f"Processed {len(processed_data)} records",
            config,
            "preprocessing"
        )

        # Step 3: Model Training
        logging.info("Training machine learning model...")
        model = train_model(processed_data)
        if model is None:
            raise ValueError("Model training failed")
        
        # Save the trained model
        model_path = os.path.join(config["model_path"], f"model_{datetime.now().strftime('%Y%m%d')}.pkl")
        pd.to_pickle(model, model_path)
        generate_report("Model Training: ML model trained and saved", config, "training")

        # Step 4: Prediction
        logging.info("Making predictions on new data sample...")
        new_data = processed_data.iloc[[-1]].drop(columns=["event"])
        prediction = predict_event(model, new_data)
        
        if prediction is not None:
            generate_report(
                f"Prediction: New data prediction result: {prediction}",
                config,
                "prediction"
            )

        # Step 5: Visualization
        logging.info("Generating visualization...")
        visualize_data(data)
        
        if config["save_visualizations"]:
            vis_path = os.path.join(
                config["visualization_path"],
                f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            )
            plt.savefig(vis_path, bbox_inches='tight', dpi=300)
            logging.info("Visualization saved to: %s", vis_path)
        
        generate_report("Visualization: Generated and saved solar activity dashboard", config, "visualization")
        
        logging.info("AI-Powered Space Weather Prediction MVP execution complete")
        return True

    except ConfigurationError as e:
        logging.error("Configuration error: %s", str(e))
        return False
    except ValueError as e:
        logging.error("Data validation error: %s", str(e))
        return False
    except Exception as e:
        logging.error("Unexpected error: %s", str(e))
        return False

if __name__ == "__main__":
    main() 