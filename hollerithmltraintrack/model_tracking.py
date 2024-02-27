# model_tracking.py
import logging
import time
from contextlib import contextmanager

import pandas as pd
from sklearn.base import clone

from .energy_tracker import EnergyTracker
from .feature_analyzer import FeatureAnalyzer

logger = logging.getLogger(__name__)


class ModelTracker:
    """
    Tracks and records details about the training process of machine learning models,
    including preprocessing feature analysis and capturing model parameters.
    """
    def __init__(self):
        self.tracked_models_info = []
        self.feature_analyzer = FeatureAnalyzer()
        self.energy_tracker = EnergyTracker()


    @contextmanager
    def track_model(self, model, X_train, y_train, preprocessor=None, num_features=None, cat_features=None):
        """
        Context manager that tracks the model training process, including timing and capturing model parameters.
        Falls back to automatic feature analysis if no preprocessor info is provided.
        """
        logger.info("Start tracking model training.")
        try:
            self.energy_tracker.start()
            start_time = time.time()
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            end_time = time.time()
            
            emissions_data = self.energy_tracker.stop_and_extract_data()
            training_duration = end_time - start_time

            if num_features is not None and cat_features is not None:
                numeric_features_count = num_features
                categorical_features_count = cat_features
            elif preprocessor:
                numeric_features_count, categorical_features_count = self.feature_analyzer.analyze_features_with_preprocessor(preprocessor)
            else:
                numeric_features_count, categorical_features_count = self.feature_analyzer.analyze_features(X_train)

            model_params_transformed = {f"model_params_{k}": v for k, v in model_clone.get_params().items()}
            tracked_info = {
                "model_type": type(model_clone).__name__,
                **model_params_transformed,
                "training_duration": training_duration,
                "numeric_features_count": numeric_features_count,
                "categorical_features_count": categorical_features_count,
            }

            if isinstance(emissions_data, dict):
                for key, value in emissions_data.items():
                    tracked_info[f"emissions_{key}"] = value

            self.tracked_models_info.append(tracked_info)

        except Exception as e:
            logger.error(f"Error during model training tracking: {e}")
        finally:
            logger.info("Model training tracking finished.")
            yield 

    def get_tracked_info(self):
        """
        Returns the information collected during model training.
        """
        return self.tracked_models_info
