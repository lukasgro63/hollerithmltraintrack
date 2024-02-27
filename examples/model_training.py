import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from hollerithmltraintrack.main import MLModelTrackerInterface

# Load the data
titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
titanic_data = pd.read_csv(titanic_url)
X = titanic_data.drop(columns=['Survived'])
y = titanic_data['Survived']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Determine the column types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_train.select_dtypes(include=['object']).columns

print("Number of numeric features:", len(numeric_features))
print("Number of categorical features:", len(categorical_features))

# Define the column transformations
preprocessor = ColumnTransformer(
    transformers=[
        ('numeric', SimpleImputer(strategy='mean'), numeric_features),
        ('categorical', OneHotEncoder(), categorical_features)
    ],
    remainder='passthrough'
)

# Process the training data
X_train_processed = preprocessor.fit_transform(X_train)

# Initialize the model
model = RandomForestClassifier(n_estimators=5, max_depth=2)

# Create the MLModelTrackerInterface instance
tracker_interface = MLModelTrackerInterface()

# Track the model training with the processed training data
# TODO: Replace "custom_file_name.csv" with the desired filename
with tracker_interface.track_training(model, X_train_processed, y_train, preprocessor= preprocessor, num_features=None, cat_features=None, filename="custom_file_name.csv"):
    model.fit(X_train_processed, y_train)

# Retrieve and print the tracked training information
tracked_info = tracker_interface.get_tracked_info()
print(tracked_info)