import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import logging

def train_model(df):
    """
    Train a machine learning model to predict space weather events.
    
    Parameters:
    df (pandas.DataFrame): Processed data including features and target column 'event'.

    Returns:
    RandomForestClassifier: Trained model.
    """
    if "event" not in df.columns:
        logging.error("Target column 'event' not found in data.")
        return None

    # Separate features and target
    X = df.drop(columns=["event"])
    y = df["event"]

    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    logging.info("Data split into training and testing sets.")

    # Initialize and train the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    logging.info("Random Forest model trained.")

    # Evaluate the model on test set
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred)
    logging.info("Model evaluation report:\n%s", report)
    print("Model Evaluation Report:\n", report)

    return model 