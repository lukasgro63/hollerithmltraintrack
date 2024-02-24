from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from hollerithmltraintrack.main import MLModelTrackerInterface

# Load Iris data and divide it into training and test data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier model
model = RandomForestClassifier(n_estimators=5, max_depth=2)

# Create an instance of MLModelTrackerInterface
tracker_interface = MLModelTrackerInterface()

# Use context manager to track the model training
with tracker_interface.model_tracker.track_model(model, X_train, y_train):
    model.fit(X_train, y_train) 

# Retrieve and output the collected training information
tracked_info = tracker_interface.get_tracked_info()
print(tracked_info)
