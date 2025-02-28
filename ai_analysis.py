import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from typing import Tuple, List, Optional

class SolarActivityPredictor:
    def __init__(self, config_path: str = "config.json"):
        self.sequence_length = 24  # 24 hours of data for prediction
        self.forecast_horizon = 12  # Predict next 12 hours
        self.n_features = 2  # Number of features (intensity, kp_index)
        self.model = None  # Will be built when data shape is known
        self.scaler = MinMaxScaler()
        self.features = ['intensity', 'kp_index']  # Define features once
        self._is_fitted = False  # Track if scaler is fitted
        
    def _build_model(self, input_shape: tuple) -> Sequential:
        """Build LSTM model for time series prediction"""
        model = Sequential()
        model.add(Input(shape=input_shape))
        model.add(LSTM(64, activation='relu', return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.forecast_horizon))
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def prepare_sequences(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        # Scale features
        if not self._is_fitted:
            scaled_data = self.scaler.fit_transform(data[self.features])
            self._is_fitted = True
        else:
            scaled_data = self.scaler.transform(data[self.features])
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length - self.forecast_horizon):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length:
                                i + self.sequence_length + self.forecast_horizon, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        logging.info(f"Input shape: {X.shape}, Output shape: {y.shape}")
        return X, y
    
    def train(self, data: pd.DataFrame, epochs: int = 50) -> dict:
        """Train the model on historical data"""
        X, y = self.prepare_sequences(data)
        
        # Build model if not already built
        if self.model is None:
            input_shape = (self.sequence_length, self.n_features)
            self.model = self._build_model(input_shape)
            logging.info(f"Built model with input shape: {input_shape}")
        
        # Split into train and validation
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return {
            'loss': history.history['loss'][-1],
            'val_loss': history.history['val_loss'][-1]
        }
    
    def predict_future(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate predictions for future solar activity"""
        if self.model is None:
            raise ValueError("Model must be trained before making predictions")
        
        if not self._is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare latest sequence
        latest_data = data.tail(self.sequence_length)
        try:
            scaled_sequence = self.scaler.transform(latest_data[self.features])
            sequence = np.expand_dims(scaled_sequence, axis=0)
            
            # Generate prediction
            prediction = self.model.predict(sequence, verbose=0)
            
            # Reshape prediction for inverse transform
            pred_sequence = np.zeros((self.forecast_horizon, self.n_features))
            pred_sequence[:, 0] = prediction[0]  # First feature is intensity
            
            # Inverse transform
            prediction_rescaled = self.scaler.inverse_transform(pred_sequence)[:, 0]
            
            # Create forecast DataFrame
            forecast_times = pd.date_range(
                start=data.index[-1],
                periods=self.forecast_horizon + 1,
                freq='h'
            )[1:]
            
            forecast_df = pd.DataFrame({
                'timestamp': forecast_times[:len(prediction_rescaled)],
                'predicted_intensity': prediction_rescaled,
                'confidence': self._calculate_confidence(prediction_rescaled)
            })
            
            return forecast_df
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            raise ValueError(f"Error making prediction: {str(e)}")
    
    def _calculate_confidence(self, predictions: np.ndarray) -> np.ndarray:
        """Calculate confidence scores for predictions"""
        # Simple confidence calculation based on prediction stability
        confidence = 1.0 - np.abs(np.diff(predictions, prepend=predictions[0])) / 2
        return np.clip(confidence, 0, 1)
    
    def generate_insights(self, data: pd.DataFrame) -> List[str]:
        """Generate AI-powered insights from the data"""
        insights = []
        
        # Analyze trends
        recent_trend = data['intensity'].tail(24).diff().mean()
        if abs(recent_trend) > 0.1:
            trend_direction = "increasing" if recent_trend > 0 else "decreasing"
            insights.append(
                f"Solar activity shows a {trend_direction} trend "
                f"over the last 24 hours."
            )
        
        # Detect patterns
        std_dev = data['intensity'].std()
        if std_dev > data['intensity'].mean() * 0.5:
            insights.append(
                "High variability detected in solar activity, "
                "suggesting potential upcoming events."
            )
        
        # Analyze correlations
        if 'kp_index' in data.columns:
            corr = data['intensity'].corr(data['kp_index'])
            if abs(corr) > 0.7:
                insights.append(
                    f"Strong {'positive' if corr > 0 else 'negative'} correlation "
                    f"between solar intensity and geomagnetic activity."
                )
        
        return insights

class AIReportGenerator:
    def __init__(self):
        self.predictor = SolarActivityPredictor()
    
    def generate_report(self, data: pd.DataFrame) -> str:
        """Generate an AI-powered analysis report"""
        try:
            # Train the model if not already trained
            if self.predictor.model is None:
                logging.info("Training model for predictions...")
                self.predictor.train(data, epochs=10)
            
            forecast = self.predictor.predict_future(data)
            
            if forecast.empty:
                raise ValueError("No forecast data generated")
            
            insights = self.predictor.generate_insights(data)
            
            # Format the report
            report = [
                "AI-Powered Solar Activity Analysis",
                "=" * 40,
                "",
                "Forecast Summary:",
                "-" * 20
            ]
            
            # Add forecast details with error handling
            try:
                max_intensity = forecast['predicted_intensity'].max()
                max_time = forecast.loc[
                    forecast['predicted_intensity'].idxmax(), 'timestamp'
                ]
                
                report.extend([
                    f"Peak Activity: {max_intensity:.2f} expected at {max_time}",
                    f"Average Confidence: {forecast['confidence'].mean():.2%}",
                    "",
                    "Predictions for next 12 hours:",
                    "-" * 20
                ])
                
                # Add hourly predictions
                for _, row in forecast.iterrows():
                    report.append(
                        f"  {row['timestamp'].strftime('%H:%M')}: "
                        f"{row['predicted_intensity']:.2f} "
                        f"(confidence: {row['confidence']:.1%})"
                    )
                
            except Exception as e:
                logging.error(f"Error formatting forecast details: {e}")
                report.append("Error processing forecast details")
            
            # Add insights and recommendations
            report.extend([
                "",
                "Key Insights:",
                "-" * 20
            ])
            
            for insight in insights:
                report.append(f"• {insight}")
            
            report.extend([
                "",
                "Recommendations:",
                "-" * 20,
                "• Monitor solar activity closely during predicted peak times",
                "• Prepare for potential impacts during high-activity periods",
                "• Update prediction models with new data regularly"
            ])
            
            return "\n".join(report)
            
        except Exception as e:
            error_msg = f"Error generating report: {str(e)}"
            logging.error(error_msg)
            return error_msg 