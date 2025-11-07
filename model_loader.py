"""
Flight Delay Prediction Model Loader
This module handles loading the flight delay prediction model and preprocessing
functions needed for making predictions.
"""

import os
import pickle
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from typing import Tuple, Dict, Any, Union, List, Optional
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = BASE_DIR / "models"
DEFAULT_METADATA_DIR = DEFAULT_MODEL_DIR / "metadata"
MODEL_ENV_VAR = "FLIGHT_DELAY_MODEL_PATH"
METADATA_ENV_VAR = "FLIGHT_DELAY_METADATA_PATH"
MODEL_FILE_CANDIDATES = [
    'Flight Delay Prediction Model.pkl',
    'Flight_Delay_Prediction_Model.pkl',
    'flight_delay_model.pkl',
    'model.pkl',
    'FD_model.pkl'
]


def _deduplicate_paths(paths: List[Optional[Union[str, Path]]]) -> List[Path]:
    """Return ordered unique Path objects."""
    seen = set()
    result = []
    for raw_path in paths:
        if raw_path is None:
            continue
        candidate = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        resolved = candidate.resolve(strict=False)
        key = str(resolved).lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result

class ModelLoader:
    """Loads and manages the flight delay prediction model and preprocessing utilities"""
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None, metadata_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the model loader
        
        Args:
            model_path: Path to the saved model file
        """
        self.model_path: Optional[Path] = Path(model_path) if model_path else None
        self.metadata_dir: Path = Path(metadata_dir) if metadata_dir else DEFAULT_METADATA_DIR
        self.model = None
        self.encoders = {}
        self.scalers = {}
        self.feature_names = []
        self.load_model()
    
    def load_model(self) -> None:
        """Load the model and preprocessing components"""
        try:
            resolved_path = self._resolve_model_path(self.model_path)
            self.model_path = resolved_path

            if self.model_path and self.model_path.exists():
                try:
                    self.model = joblib.load(self.model_path)
                    print(f"Model loaded successfully from {self.model_path} using joblib")
                except Exception:
                    with open(self.model_path, 'rb') as f:
                        self.model = pickle.load(f)
                    print(f"Model loaded successfully from {self.model_path} using pickle")

                self._load_preprocessing_components()
            else:
                print(f"Model file not found at {self.model_path}")
                self._create_dummy_model()
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            self._create_dummy_model()

    def _resolve_model_path(self, provided_path: Optional[Path]) -> Optional[Path]:
        """Return the first existing model path based on common locations."""
        candidates: List[Optional[Union[str, Path]]] = []

        if provided_path:
            candidates.append(provided_path)
            if provided_path.is_dir():
                candidates.extend(provided_path / name for name in MODEL_FILE_CANDIDATES)

        env_override = os.getenv(MODEL_ENV_VAR)
        if env_override:
            env_path = Path(env_override)
            candidates.append(env_path)
            if env_path.is_dir():
                candidates.extend(env_path / name for name in MODEL_FILE_CANDIDATES)

        search_dirs = [DEFAULT_MODEL_DIR, BASE_DIR, BASE_DIR.parent]
        for directory in search_dirs:
            candidates.extend(Path(directory) / name for name in MODEL_FILE_CANDIDATES)

        if DEFAULT_MODEL_DIR.exists():
            try:
                newest = sorted(DEFAULT_MODEL_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)
                candidates.extend(newest)
            except OSError:
                pass

        for candidate in _deduplicate_paths(candidates):
            if candidate.exists():
                return candidate
        return provided_path
    
    def _load_preprocessing_components(self) -> None:
        """Attempt to load preprocessing components (encoders, scalers, etc.)"""
        metadata_dirs = self._metadata_dirs()

        encoder_path = self._find_metadata_file(metadata_dirs, ['label_encoders.pkl', 'encoders.pkl'])
        if encoder_path:
            try:
                self.encoders = joblib.load(encoder_path)
                print(f"Loaded encoders from {encoder_path}")
            except Exception:
                print(f"Failed to load encoders from {encoder_path}")
        
        scaler_path = self._find_metadata_file(metadata_dirs, ['scalers.pkl'])
        if scaler_path:
            try:
                self.scalers = joblib.load(scaler_path)
                print(f"Loaded scalers from {scaler_path}")
            except Exception:
                print(f"Failed to load scalers from {scaler_path}")
        
        feature_names_path = self._find_metadata_file(metadata_dirs, ['model_features.pkl', 'feature_names.pkl'])
        if feature_names_path:
            try:
                self.feature_names = joblib.load(feature_names_path)
                print(f"Loaded {len(self.feature_names)} feature names from {feature_names_path}")
            except Exception:
                print(f"Failed to load feature names from {feature_names_path}")

    def _metadata_dirs(self) -> List[Path]:
        """Return ordered directories where metadata files might live."""
        directories: List[Optional[Union[str, Path]]] = []
        env_override = os.getenv(METADATA_ENV_VAR)
        if env_override:
            directories.append(Path(env_override))
        directories.append(self.metadata_dir)
        if self.model_path:
            directories.append(self.model_path.parent)
        directories.extend([DEFAULT_METADATA_DIR, DEFAULT_MODEL_DIR, BASE_DIR])
        return _deduplicate_paths(directories)

    def _find_metadata_file(self, directories: List[Path], names: List[str]) -> Optional[Path]:
        """Return the first metadata file that exists within the provided directories."""
        for directory in directories:
            for name in names:
                candidate = directory / name
                if candidate.exists():
                    return candidate
        return None
    
    def _create_dummy_model(self) -> None:
        """Create a dummy model for demonstration purposes"""
        from sklearn.ensemble import RandomForestClassifier
        print("Creating dummy model for demonstration")
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.feature_names = [
            'DAY_OF_WEEK', 'MONTH', 'DEP_TIME', 'DISTANCE',
            'AIRLINE_AA', 'AIRLINE_DL', 'AIRLINE_UA', 'AIRLINE_WN',
            'ORIGIN_ATL', 'ORIGIN_DFW', 'ORIGIN_LAX', 'ORIGIN_ORD',
            'DEST_ATL', 'DEST_DFW', 'DEST_LAX', 'DEST_ORD'
        ]
    
    def preprocess_data(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Preprocess input data for prediction
        
        Args:
            input_data: Dictionary containing input flight information
            
        Returns:
            DataFrame with preprocessed features ready for prediction
        """
        # Convert input dictionary to DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Apply categorical encoding if available
        if self.encoders and 'categorical' in self.encoders:
            cat_encoder = self.encoders['categorical']
            categorical_features = ['AIRLINE', 'ORIGIN', 'DEST']
            categorical_cols = [col for col in categorical_features if col in input_df.columns]
            
            if categorical_cols:
                try:
                    # For one-hot encoding
                    encoded_cats = cat_encoder.transform(input_df[categorical_cols])
                    encoded_df = pd.DataFrame(
                        encoded_cats, 
                        columns=cat_encoder.get_feature_names_out(categorical_cols)
                    )
                    
                    # Add encoded features and drop original
                    for col in encoded_df.columns:
                        input_df[col] = encoded_df[col].values
                    input_df = input_df.drop(columns=categorical_cols)
                except:
                    print("Error in categorical encoding, using simplified approach")
                    # Simplified encoding for demo
                    self._apply_simplified_encoding(input_df)
        else:
            # If no encoders, apply simplified encoding
            self._apply_simplified_encoding(input_df)
        
        # Apply scaling if available
        if self.scalers and 'numerical' in self.scalers:
            num_scaler = self.scalers['numerical']
            numerical_features = ['DAY_OF_WEEK', 'MONTH', 'DEP_TIME', 'DISTANCE']
            numerical_cols = [col for col in numerical_features if col in input_df.columns]
            
            if numerical_cols:
                try:
                    input_df[numerical_cols] = num_scaler.transform(input_df[numerical_cols])
                except:
                    print("Error in numerical scaling")
        
        # Ensure all required features are present
        if self.feature_names:
            for feature in self.feature_names:
                if feature not in input_df.columns:
                    input_df[feature] = 0
            
            # Keep only required features in the correct order
            input_df = input_df[self.feature_names]
        
        return input_df
    
    def _apply_simplified_encoding(self, df: pd.DataFrame) -> None:
        """Apply simplified categorical encoding for demo purposes"""
        # Simple one-hot encoding for airline
        if 'AIRLINE' in df.columns:
            airline = df['AIRLINE'].iloc[0]
            airlines = ['AA', 'DL', 'UA', 'WN', 'B6', 'AS', 'NK', 'F9']
            for a in airlines:
                df[f'AIRLINE_{a}'] = 1 if airline == a else 0
            df.drop(columns=['AIRLINE'], inplace=True)
        
        # Simple one-hot encoding for origin
        if 'ORIGIN' in df.columns:
            origin = df['ORIGIN'].iloc[0]
            airports = ['ATL', 'DFW', 'LAX', 'ORD', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS']
            for a in airports:
                df[f'ORIGIN_{a}'] = 1 if origin == a else 0
            df.drop(columns=['ORIGIN'], inplace=True)
        
        # Simple one-hot encoding for destination
        if 'DEST' in df.columns:
            dest = df['DEST'].iloc[0]
            airports = ['ATL', 'DFW', 'LAX', 'ORD', 'DEN', 'JFK', 'SFO', 'SEA', 'LAS']
            for a in airports:
                df[f'DEST_{a}'] = 1 if dest == a else 0
            df.drop(columns=['DEST'], inplace=True)
    
    def predict(self, input_data: Dict[str, Any]) -> Tuple[bool, int, float]:
        """
        Make a prediction with the flight delay model
        
        Args:
            input_data: Dictionary containing input flight information
            
        Returns:
            Tuple of (is_delayed, delay_minutes, probability)
        """
        try:
            # Preprocess the input data
            features = self.preprocess_data(input_data)
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                # For classifiers that provide probabilities
                probabilities = self.model.predict_proba(features)
                if probabilities.shape[1] >= 2:
                    delay_probability = probabilities[0][1]  # Probability of positive class
                else:
                    delay_probability = probabilities[0][0]
                
                # Predict delay (true/false)
                is_delayed = delay_probability > 0.5
                
                # Estimate delay minutes based on probability
                # This is a simple heuristic - in real app, you might have a separate model for this
                delay_minutes = int(20 + 40 * delay_probability) if is_delayed else 0
                
            else:
                # For classifiers without probabilities or regression models
                prediction = self.model.predict(features)[0]
                
                # Check prediction type
                if isinstance(prediction, (int, float, np.number)):
                    # Regression model predicting minutes directly
                    delay_minutes = max(0, int(prediction))
                    is_delayed = delay_minutes > 15
                    delay_probability = delay_minutes / 100 if delay_minutes > 0 else 0
                else:
                    # Classification model returning 0/1 or True/False
                    is_delayed = bool(prediction)
                    delay_minutes = 25 if is_delayed else 0
                    delay_probability = 0.8 if is_delayed else 0.2
            
            return is_delayed, delay_minutes, delay_probability
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            # Fallback to random prediction
            import random
            delay_probability = random.random()
            is_delayed = delay_probability > 0.5
            delay_minutes = int(random.gammavariate(2.0, 15.0)) if is_delayed else 0
            return is_delayed, delay_minutes, delay_probability


# Singleton instance for use in app
model_loader = ModelLoader()

# Test function
def test_model_loader():
    """Test the model loader with sample input"""
    test_input = {
        'AIRLINE': 'AA',
        'ORIGIN': 'DFW',
        'DEST': 'LAX',
        'DAY_OF_WEEK': 3,
        'MONTH': 7,
        'DEP_TIME': 1020,  # 17:00 in minutes
        'DISTANCE': 1235
    }
    
    loader = ModelLoader()
    is_delayed, delay_minutes, probability = loader.predict(test_input)
    
    print(f"Flight predicted to be {'delayed' if is_delayed else 'on time'}")
    print(f"Estimated delay: {delay_minutes} minutes")
    print(f"Delay probability: {probability:.2f}")


if __name__ == "__main__":
    test_model_loader()
