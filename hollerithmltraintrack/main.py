from .model_tracking import ModelTracker


class MLModelTrackerInterface:
    def __init__(self):
        self.model_tracker = ModelTracker()

    def track_training(self, model, X_train, y_train, preprocessor):
        """
        Starts the tracking process and returns the tracking result.
        Utilizes the track_model method from ModelTracker directly.
        """
        return self.model_tracker.track_model(model, X_train, y_train, preprocessor)

    def get_tracked_info(self):
        """
        Retrieves information about the tracked training processes.
        """
        return self.model_tracker.get_tracked_info()
