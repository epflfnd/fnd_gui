import numpy as np

from itertools import chain
from scipy import sparse
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin


# Custom Transformer that extracts columns passed as argument to its constructor
class ColumnSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, col_name="text"):
        self.col_name = col_name

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        return X[self.col_name]


# Custom Transformer that let the possibility to not normalize the input in our gridsearch
class FrequencyNormalizer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, norm=None):
        self.norm = norm

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

        # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.norm:
            normalizer = preprocessing.Normalizer(norm=self.norm)
            return normalizer.fit_transform(X)
        else:
            return X


# Custom Transformer that extracts columns passed as argument to its constructor
class CfgSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, leaves=None):
        self.leaves = leaves

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.leaves:
            X = X.map(lambda rules_dict: rules_dict[self.leaves])
        else:
            X = X.map(lambda rules_dict: list(chain(*rules_dict.values())))
        return X


# Custom Transformer that extracts columns passed as argument to its constructor
class CfgSelector2(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, leaves=None):
        self.leaves = leaves

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.leaves:
            X = X.map(lambda rules_dict: rules_dict[self.leaves])
        else:
            X = X.map(lambda rules_dict: rules_dict["lexical"] + rules_dict["non_lexical"])
        return X


class MaxFeatSelector(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, max_f=None, binary=False):
        self.max_f = max_f
        self.binary = binary

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        X = X.copy()
        max_f_id = np.squeeze(np.asarray(X.sum(axis=0).argsort(axis=1)))[::-1][:self.max_f]
        if self.binary:
            X[X > 0] = 1
        return X[:, max_f_id]

    
class BinaryTransformer(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, binary=False):
        self.binary = binary

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if self.binary:
            X = X.copy()
            X[X > 0] = 1
        return X


class DictVectorizerCustom(BaseEstimator, TransformerMixin):
    # Class Constructor
    def __init__(self, cats=None):
        self.cats = cats
        self.vocabulary = cats

    # Return self nothing else to do here
    def fit(self, X, y=None):
        return self

    # Method that describes what we need this transformer to do
    def transform(self, X, y=None):
        if not self.cats:
            self.vocabulary = list(X.iloc[0].keys())
        X_array = X.map(lambda d: self._select_cat(d, self.cats))
        return sparse.csr_matrix(np.stack(X_array))
        
    def get_feature_names(self):
        return self.vocabulary

    def _select_cat(self, dico, cats=None):
        if not cats:
            vect = list(dico.values())
        else:
            vect = [dico[c] for c in cats]
        return vect