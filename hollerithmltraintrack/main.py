# main.py

from .model_tracking import ModelTracker


class MLModelTrackerInterface:
    """
    A class to track and analyze the training process data of ML models
    """

    def __init__(self):
        self.model_tracker = ModelTracker()
    
    def track_training(self, model, X_train, y_train, training_function, *args, **kwargs):
        """
        Enables the tracking of a training process of a model

        params:
            - model: the model to be trained
            - X_train: the training data
            - y_train: the target values
            - training_function: the function that trains the model
            - args, kwargs: additional arguments and keyword arguments for the training function
        """
        with self.model_tracker.track_model(model, X_train, y_train):
            training_function(model, X_train, y_train, *args, **kwargs)

    def get_tracked_info(self):
        return self.model_tracker.get_tracked_info()
        

if __name__ == "__main__":
    tracker = MLModelTrackerInterface()
    