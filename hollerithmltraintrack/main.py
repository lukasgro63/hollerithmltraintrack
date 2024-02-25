import logging
from contextlib import contextmanager

from .csv_exporter import CSVExporter
from .model_tracking import ModelTracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MLModelTrackerInterface:
    def __init__(self, default_filename="model_tracking_data.csv"):
        self.model_tracker = ModelTracker()
        self.default_filename = default_filename

    @contextmanager
    def track_training(self, model, X_train, y_train, preprocessor, filename=None):
        """
        Context manager that starts the tracking process, captures the tracking result,
        and automatically exports it to a CSV file.
        """
        csv_filename = filename if filename else self.default_filename
        self.csv_exporter = CSVExporter(csv_filename)

        with self.model_tracker.track_model(model, X_train, y_train, preprocessor) as _:
            yield

        tracked_info = self.model_tracker.get_tracked_info()
        self.csv_exporter.export_to_csv(tracked_info)

    def get_tracked_info(self):
        """
        Retrieves information about the tracked training processes.
        """
        return self.model_tracker.get_tracked_info()
