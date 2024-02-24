# hollerithmltraintrack/model_tracking.py
import time
from contextlib import contextmanager


class ModelTracker:
    def __init__(self):
        self.tracked_models_info = []

    @contextmanager
    def track_model(self, model, X_train, y_train):
        """
        records the model, training data

        params:
            - model: the model to be tracked
            - X_train: the training data that was used to train the model
            - y_train: the target values that were used to train the model
        """
        start_time = time.time()
        model_type = type(model).__name__
        features_count = X_train.shape[1]
        samples_count = X_train.shape[0]
        model_params = model.get_params()

        yield
        
        end_time = time.time()
        training_duration = end_time - start_time

        self.tracked_models_info.append({
            "model_type": model_type,
            "features_count": features_count,
            "samples_count": samples_count,
            "model_params": model_params,
            "training_duration": training_duration
        })

        print(f"Tracking finished for model: {model_type}")

    def get_tracked_info(self):
        return self.tracked_models_info

