import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)

class CSVExporter:
    """
    Exporting model tracking data to a CSV file.
    """

    def __init__(self, filename="model_tracking_data.csv"):
        """
        Initializes the CSV exporter with the filename where the data will be saved.
        """
        self.filename = filename

    def export_to_csv(self, tracked_data):
        """
        Exports the tracked model training data to a CSV file.
        """
        try:
            df = pd.DataFrame(tracked_data)
            full_path = os.path.join(os.getcwd(), self.filename)
            df.to_csv(full_path, index=False)
            logger.info(f"Data successfully exported to {full_path}")
        except Exception as e:
            logger.error(f"Error when exporting the data: {e}")