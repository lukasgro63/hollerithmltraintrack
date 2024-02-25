# HollerithMLTrainTrack
 is a Python library designed to track and analyze machine learning model training sessions, integrating key metrics such as training duration, feature analysis, and environmental impact metrics through CodeCarbon integration. It aims to provide developers and researchers with insights into their model's performance and its carbon footprint, enhancing transparency and accountability in ML projects.

## Installation

Install HollerithMLTrainTrack
 with pip:

```bash
pip install hollerithmltraintrack
```

## Quick Start

To get started with MLModelTracker, follow this simple example:

```python
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
with tracker_interface.track_training(model, X_train_processed, y_train, preprocessor, filename="custom_file_name.csv"):
    model.fit(X_train_processed, y_train)

# Retrieve and print the tracked training information
tracked_info = tracker_interface.get_tracked_info()
print(tracked_info)
```

This example demonstrates how to track the training of a RandomForestClassifier on the Iris dataset. More detailed examples can be found in the `examples` folder.

## Features

- **Model Training Tracking:** Capture detailed information about each training session, including model parameters, training duration, and feature counts.
- **CodeCarbon Integration:** Automatically track and report the carbon emissions associated with model training, leveraging the CodeCarbon library.
- **CSV Export:** Export collected data into a CSV file for further analysis or documentation purposes.

## Advanced Usage

*Docs to be added*

## Limitations and Known Issues

- Currently supports only sklearn models. Support for other frameworks is planned for future releases.
- The feature extraction only works when using the sklearn `ColumnTransformer`
- CodeCarbon integration relies on external services for emissions data, which may be subject to availability.

## Contributing

Contributions are welcome! If you'd like to contribute, please:
- Fork the repository.
- Create a new branch for your feature.
- Submit a pull request.

Please refer to CONTRIBUTING.md for detailed contribution guidelines.

## License

MLModelTracker is released under the MIT License. See the LICENSE file for more details.

## Contact

For support or queries, please reach out to us at lukasgro63@gmail.com.

Feel free to report any issues or suggestions!