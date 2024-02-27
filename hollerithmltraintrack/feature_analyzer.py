# feature_analyzer.py
import logging
from collections import defaultdict

import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    def __init__(self, preprocessor=None):
        # Der Preprocessor wird verwendet, um Spaltennamen aus einem csr_matrix zu extrahieren, falls notwendig.
        self.preprocessor = preprocessor
        self.feature_info = defaultdict(lambda: {'is_categorical': False, 'original_name': None, 'categories': []})

    def analyze_features(self, X_train):
        # Überprüfe den Typ von X_train und handle entsprechend
        if self.preprocessor and hasattr(X_train, 'toarray'):
            # Rekonstruiere Spaltennamen basierend auf dem Preprocessor, falls X_train eine csr_matrix ist
            column_names = self.preprocessor.get_feature_names_out()
            X_train = pd.DataFrame(X_train.toarray(), columns=column_names)
        elif isinstance(X_train, pd.DataFrame):
            # Wenn X_train bereits ein DataFrame ist, nutze die Spaltennamen direkt
            column_names = X_train.columns
        else:
            # Logge einen Fehler, falls X_train nicht unterstützt wird
            print("Unsupported data format. Please provide a DataFrame or csr_matrix with a preprocessor.")
            return None

        # Identifiziere und verifiziere One-Hot-kodierte Features
        one_hot_encoded_features = self.identify_one_hot_encoded_columns(column_names)
        verified_one_hot_features = self.verify_one_hot_encoding(X_train, one_hot_encoded_features)

        # Berechne die Anzahl der numerischen und kategorischen Features
        num_features_count = len(column_names) - sum(len(cols) for cols in verified_one_hot_features.values())
        cat_features_count = len(verified_one_hot_features)

        return num_features_count, cat_features_count

    def identify_one_hot_encoded_columns(self, column_names):
        # Identifiziert potenziell One-Hot-kodierte Spalten basierend auf den Spaltennamen
        one_hot_candidates = defaultdict(list)
        for col in column_names:
            if '_' in col:
                original_name, category = col.rsplit('_', 1)
                one_hot_candidates[original_name].append(col)
                # Aktualisiere die Feature-Information
                self.feature_info[original_name]['is_categorical'] = True
                self.feature_info[original_name]['original_name'] = original_name
                self.feature_info[original_name]['categories'].append(category)
        return one_hot_candidates

    def verify_one_hot_encoding(self, X_train, one_hot_candidates):
        # Verifiziert, ob identifizierte Spalten korrekt One-Hot-kodiert sind
        verified_one_hot_features = {}
        for original_name, cols in one_hot_candidates.items():
            if all(X_train[cols].sum(axis=1).isin([0, 1])):
                verified_one_hot_features[original_name] = cols
        return verified_one_hot_features

    def analyze_features_with_preprocessor(self, preprocessor):
        """
        Analysiert Features basierend auf dem gegebenen Preprocessor.
        """
        try:
            numeric_features_count = len(preprocessor.transformers_[0][2])
            categorical_features_count = len(preprocessor.transformers_[1][2])
            return numeric_features_count, categorical_features_count
        except Exception as e:
            logger.error(f"Error analyzing features with preprocessor: {e}")
            return 0, 0


