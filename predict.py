import pandas as pd
import logging

def predict_event(model, new_data):
    """
    Use the trained model to predict space weather events on new data.
    
    Parameters:
    model: Trained machine learning model.
    new_data (pandas.DataFrame): New data in the same format as training features.
    
    Returns:
    The model's predictions.
    """
    try:
        # Ensure new_data is in a DataFrame format
        if not isinstance(new_data, pd.DataFrame):
            new_data = pd.DataFrame(new_data)
        
        predictions = model.predict(new_data)
        logging.info("Predictions made on new data.")
        return predictions
    except Exception as e:
        logging.error("Error during prediction: %s", e)
        return None 