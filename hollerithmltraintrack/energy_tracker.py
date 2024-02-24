# energy_tracker.py

from codecarbon import EmissionsTracker


class EnergyTracker:
    def __init__(self, output_file="emissions.csv"):
        self.tracker = EmissionsTracker(output_file=output_file)

    def start(self):
        self.tracker.start()

    def stop(self):
        emissions_data = self.tracker.stop()
        return emissions_data