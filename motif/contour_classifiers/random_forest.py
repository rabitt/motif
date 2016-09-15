"""Random Forest contour classifier.
"""
from __future__ import print_function
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.grid_search import RandomizedSearchCV
from sklearn.utils import shuffle
from scipy.stats import randint as sp_randint

from motif.core import ContourClassifier


class RandomForest(ContourClassifier):
    '''Random Forest contour classifier.

    Attributes
    ----------
    n_estimators : int
        Number of trees in the forest
    n_jobs : int
        Number of cores to use. -1 uses maximum availalbe
    class_weight : str
        How to set class weights.
    max_features : int or None
        The maximum number of features that can be used in a single branch.
    max_param : int
        Maximum depth value to sweep
    param_step : int
        Step size in parameter sweep
    clf : sklearn.ensemble.RandomForestClassifier
        Classifier
    max_depth : int
        The max_depth parameter chosen by cross validation.

    '''
    def __init__(self, n_estimators=50, n_jobs=-1, class_weight='balanced',
                 n_iter_search=100):
        '''
        Parameters
        ----------
        n_estimators : int
            Number of trees in the forest
        n_jobs : int
            Number of cores to use. -1 uses maximum availalbe
        class_weight : str
            How to set class weights.

        '''
        ContourClassifier.__init__(self)

        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.class_weight = class_weight
        self.n_iter_search = n_iter_search
        self.clf = None

    def predict(self, X):
        """ Compute probability predictions.

        Parameters
        ----------
        X : np.array [n_samples, n_features]
            Features.

        Returns
        -------
        p : np.array [n_samples]
            predicted probabilities
        """
        if self.clf is None:
            raise ReferenceError(
                "fit must be called before predict can be called"
            )
        p_train = self.clf.predict_proba(X)[:, 1]
        return p_train

    def fit(self, X, Y):
        """ Train classifier.

        Parameters
        ----------
        X : np.array [n_samples, n_features]
            Training features.
        Y : np.array [n_samples]
            Training labels

        """
        x_shuffle, y_shuffle = shuffle(X, Y)
        # self._cross_val_sweep(x_shuffle, y_shuffle)
        clf_cv = RFC(n_estimators=self.n_estimators, n_jobs=self.n_jobs,
                     class_weight=self.class_weight)
        param_dist = {
            "max_depth": sp_randint(1, 101),
            "max_features": [None, 'auto', 'sqrt', 'log2'],
            "min_samples_split": sp_randint(1, 11),
            "min_samples_leaf": sp_randint(1, 11),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }

        random_search = RandomizedSearchCV(
            clf_cv, param_distributions=param_dist, refit=True,
            n_iter=self.n_iter_search, scoring='f1_weighted'
        )
        random_search.fit(x_shuffle, y_shuffle)
        self.clf = random_search.best_estimator_

    @property
    def threshold(self):
        """ The threshold determining the positive class.

        Returns
        -------
        threshold : flaot
            melodiness scores
        """
        return 0.5

    @classmethod
    def get_id(cls):
        """ The ContourClassifier identifier

        Returns
        -------
        id : string
            class identifier
        """
        return 'random_forest'
