# energy_tracker.py
import os

import pandas as pd
from codecarbon import EmissionsTracker


class EnergyTracker:
    def __init__(self, temp_file="temp_emissions.csv"):
        self.temp_file = temp_file
        self.tracker = EmissionsTracker(output_file=self.temp_file)

    def start(self):
        self.tracker.start()


    def stop_and_extract_data(self):
        self.tracker.stop()
        if os.path.exists(self.temp_file) and os.path.getsize(self.temp_file) > 0:
            data = pd.read_csv(self.temp_file)
            os.remove(self.temp_file)
            # Conversion of the DataFrame into a dictionary
            emissions_data_dict = data.iloc[0].to_dict()
            return emissions_data_dict
        else:
            print("Temporary emission file is empty or does not exist.")
            return {}