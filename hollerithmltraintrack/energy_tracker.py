# energy_tracker.py
import logging
import os

import pandas as pd
from codecarbon import EmissionsTracker

logger = logging.getLogger(__name__)


class EnergyTracker:
    def __init__(self, temp_file="temp_emissions.csv"):
        self.temp_file = temp_file
        self.tracker = EmissionsTracker(output_file=self.temp_file)

    def start(self):
        self.tracker.start()

    def stop_and_extract_data(self):
        self.tracker.stop()
        if os.path.exists(self.temp_file):
            try:
                data = pd.read_csv(self.temp_file)
                os.remove(self.temp_file)
                return data.to_dict(orient='records')[0]
            except Exception as e:
                logger.error(f"FError when reading the emission data: {e}")
                return {}
        else:
            logger.warning("Temporary emission file from code carbon not found.")
            return {}