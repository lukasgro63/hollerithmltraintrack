import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from hollerithmltraintrack.main import MLModelTrackerInterface

# Load the data
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(titanic_url)
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

# Convert categorical variables to dummy variables
X_dummies = pd.get_dummies(X, drop_first=True)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_dummies, y, test_size=0.2, random_state=42)

# Impute missing values in numeric columns
# First, find numeric columns because pd.get_dummies only applies to object or category types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
imputer = SimpleImputer(strategy='mean')
X_train_imputed = X_train.copy()
X_train_imputed[numeric_features] = imputer.fit_transform(X_train[numeric_features])

# Initialize the model
model = RandomForestClassifier(n_estimators=5, max_depth=2)

# Create the MLModelTrackerInterface instance
tracker_interface = MLModelTrackerInterface()

# Track the model training with the processed training data
# Note: Here, preprocessor=None, num_features and cat_features are not required since we manually process features
with tracker_interface.track_training(model, X_train_imputed, y_train, preprocessor=None, num_features=None, cat_features=None, filename="custom_file_name.csv"):
    model.fit(X_train_imputed, y_train)

# Retrieve and print the tracked training information
tracked_info = tracker_interface.get_tracked_info()
print(tracked_info)
