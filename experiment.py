import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.base import clone
from scipy.stats import ttest_ind
from exposing import EE
from tqdm import tqdm
from RS import RS

n_est = 100
base_n_features = 1500

data = np.load("data/cv.npz")
X_, y = data["X"], data["y"]

for n_features in [1000]:
    subspace_size = np.sqrt(n_features * 2).astype(int)
    print("\nsubspace_size", subspace_size, "n_est", n_est, "n_features", n_features)
    base_estimator = DecisionTreeClassifier()
    clfs = {
        " DTC": DecisionTreeClassifier(),
        " sRS": RS(
            random_state=0,
            n_estimators=n_est,
            subspace_size=subspace_size,
            fuser="s",
            base_estimator=base_estimator,
        ),
        "swRS": RS(
            random_state=0,
            n_estimators=n_est,
            subspace_size=subspace_size,
            fuser="sw",
            base_estimator=base_estimator,
        ),
        " vRS": RS(
            random_state=0,
            n_estimators=n_est,
            subspace_size=subspace_size,
            fuser="v",
            base_estimator=base_estimator,
        ),
        "vwRS": RS(
            random_state=0,
            n_estimators=n_est,
            subspace_size=subspace_size,
            fuser="vw",
            base_estimator=base_estimator,
        ),
    }
    print([clfna for clfna in clfs])

    n_splits = 5
    alpha = 0.05

    X = X_[:, np.r_[:n_features, base_n_features : base_n_features + n_features]]

    scores = np.zeros((len(clfs), n_splits))
    skf = StratifiedKFold(n_splits=n_splits, random_state=0)
    for fold, (train, test) in tqdm(
        enumerate(skf.split(X, y)), ascii=True, total=n_splits
    ):
        for cid, clfn in enumerate(clfs):
            clf = clone(clfs[clfn])
            clf.fit(X[train], y[train])
            y_pred = clf.predict(X[test])

            score = accuracy_score(y[test], y_pred)
            bac = balanced_accuracy_score(y[test], y_pred)
            # print(fold, clfn, "%.3f" % score, "%.3f" % bac)

            scores[cid, fold] = score

    mean_scores = np.mean(scores, axis=1)

    firsts = []
    seconds = []
    for a, clfna in enumerate(clfs):
        signs = []
        ids = []
        for b, clfnb in enumerate(clfs):
            if a == b:
                continue
            test = ttest_ind(scores[a], scores[b])
            sign = "=" if test.pvalue > alpha else (">" if test.statistic > 0 else "<")
            # print(b + 1, sign, clfnb)
            if sign == "=":
                ids.append(b + 1)
                signs.append(sign)
            if sign == ">":
                ids.append(b + 1)
                signs.append(sign)

        first = "%.3f" % mean_scores[a]
        second = ", ".join(
            [
                "{%s %i}" % ("\\bfseries" if signs[i] == ">" else "", n)
                for i, n in enumerate(ids)
            ]
        )

        firsts.append(first)
        seconds.append(second)

    print(" & ".join(firsts), "\\\\")
    print(" & ".join(seconds), "\\\\")
