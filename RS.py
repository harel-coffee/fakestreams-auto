from sklearn.base import clone, ClassifierMixin, BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tqdm import tqdm
from imblearn.under_sampling import RandomUnderSampler

class RS(ClassifierMixin, BaseEstimator):
    def __init__(self, subspace_size=3, n_estimators=10,
                 base_estimator=DecisionTreeClassifier(),
                 random_state=0, fuser='v'):
        self.subspace_size = subspace_size
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.random_state = random_state
        self.fuser = fuser
        #print(self)

    def fit(self, X, y):
        # RUS
        rus = RandomUnderSampler(random_state=42)
        X_res, y_res = rus.fit_resample(X, y)

        #
        np.random.seed(self.random_state)
        n_features = X.shape[1]

        # Get subset of subspaces
        self.subspaces = np.random.randint(0, n_features,
                                           (self.n_estimators,
                                            self.subspace_size))

        #print(self.subspaces, self.subspaces.shape)
        #exit()

        # Train ensemble
        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        self.ensemble = []
        for subspace in tqdm(self.subspaces, ascii=True):
            clf = clone(self.base_estimator).fit(X_train[:, subspace],
                                                            y_train)
            self.ensemble.append(clf)

        """
        self.ensemble = [clone(self.base_estimator).fit(X_train[:, subspace],
                                                        y_train)
                         for subspace in self.subspaces]
        """

        # Check qualities
        self.weights = np.zeros(self.n_estimators)
        for i, clf in enumerate(self.ensemble):
            subspace = self.subspaces[i]
            y_pred = clf.predict(X_test[:, subspace])
            score = accuracy_score(y_test, y_pred)
            self.weights[i] = score

    def predict(self, X):
        if 'v' in self.fuser:
            votes = np.array([clf.predict(X[:, self.subspaces[i]])
                                 for i, clf in enumerate(self.ensemble)])
            if 'w' in self.fuser:
                votes[votes == 0] = -1
                votes = self.weights[:, None] * votes
                y_pred = (np.sum(votes, axis=0) > 0).astype(int)
                return y_pred
            else:
                y_pred = (np.sum(votes, axis=0) > (self.n_estimators / 2)).astype(int)

                return y_pred
        elif 's' in self.fuser:
            supports = np.array([clf.predict_proba(X[:, self.subspaces[i]])
                                 for i, clf in enumerate(self.ensemble)])

            if 'w' in self.fuser:
                supports = self.weights[:, None, None] * supports

            averaged_support = np.mean(supports, axis=0)
            y_pred = np.argmax(averaged_support, axis=1)

            return y_pred
