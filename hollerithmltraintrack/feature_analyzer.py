# feature_analyzer.py
import logging
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)

class FeatureAnalyzer:
    def __init__(self):
        pass

    def analyze_features(self, df):
        one_hot_candidates = self.identify_one_hot_encoded_columns(df)
        verified_one_hot_features = self.verify_one_hot_encoding(df, one_hot_candidates)

        cat_features_count = len(verified_one_hot_features)
        num_features_count = len(df.columns) - sum(len(cols) for cols in verified_one_hot_features.values()) + cat_features_count

        return num_features_count, cat_features_count

    def identify_one_hot_encoded_columns(self, df):
        """
        Identifies potential one-hot-encoded columns based on the column names.
        """
        one_hot_candidates = defaultdict(list)
        for col in df.columns:
            if '_' in col:
                prefix = col.split('_')[0]
                one_hot_candidates[prefix].append(col)
        return one_hot_candidates
        
    def verify_one_hot_encoding(self, df, one_hot_candidates):
        """
        Verifies if the identified one-hot-encoded columns are actually one-hot-encoded.
        """
        verified_one_hot_features = {}
        for prefix, cols in one_hot_candidates.items():
            if all(df[cols].sum(axis=1).isin([0, 1])):
                verified_one_hot_features[prefix] = cols
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


