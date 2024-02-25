# model_tracking.py
import time
from contextlib import contextmanager

import pandas as pd
from sklearn.base import clone

from .energy_tracker import EnergyTracker


class ModelTracker:
    """
    Tracks and records details about the training process of machine learning models,
    including preprocessing feature analysis and capturing model parameters.
    """
    def __init__(self):
        self.tracked_models_info = []
        self.energy_tracker = EnergyTracker()

    def analyze_features(self, preprocessor):
        """
        Analyzes and counts numeric and categorical features based on the preprocessor configuration.
        """
        numeric_features_count = len(preprocessor.transformers_[0][2])
        categorical_features_count = len(preprocessor.transformers_[1][2])
        return numeric_features_count, categorical_features_count

    @contextmanager
    def track_model(self, model, X_train, y_train, preprocessor):
        """
        Context manager that tracks the model training process, including timing and capturing model parameters.
        """
        self.energy_tracker.start()
        start_time = time.time()
        model_clone = clone(model)
        model_clone.fit(X_train, y_train)
        end_time = time.time()
        
        emissions_data = self.energy_tracker.stop_and_extract_data()

        training_duration = end_time - start_time
        numeric_features_count, categorical_features_count = self.analyze_features(preprocessor)
        model_params_transformed = {f"model_params_{k}": v for k, v in model_clone.get_params().items()}
        
        tracked_info = {
            "model_type": type(model_clone).__name__,
            **model_params_transformed,
            "training_duration": training_duration,
            "numeric_features_count": numeric_features_count,
            "categorical_features_count": categorical_features_count,
        }

        # Add emissions data to the tracked information
        if isinstance(emissions_data, dict):
            for key, value in emissions_data.items():
                tracked_info[f"emissions_{key}"] = value

        self.tracked_models_info.append(tracked_info)

        try:
            yield
        finally:
            pass

    def get_tracked_info(self):
        """
        Returns the information collected during model training.
        """
        return self.tracked_models_info
