import numpy as np
import strlearn as sl
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from skmultiflow.trees import HoeffdingTree
from sklearn.feature_selection import SelectKBest, chi2

BASE_CV = 1000

methods = ["GNB", "MLP", "HT"]
# methods = ["GNB"]


class StreamFromFile:
    def __init__(
        self,
        filename,
        chunk_size=250,
        n_components=4,
        n_chunks=None,
        method=None,
        handicap=1000,
    ):
        self.handicap = handicap
        self.filename = filename
        self.data = np.load("data/cv.npz")
        self.X, self.y = self.data["X"], self.data["y"]
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        self.n_components = n_components

        if method == "PCA":
            self.pca = PCA(self.n_components).fit(self.X[: self.handicap, :])
            self.X = self.pca.transform(self.X)
        elif method == "CV":
            wordsum = np.sum(self.X[: self.handicap, :], axis=0)
            feature_sort = np.argsort(-wordsum)

            self.X = self.X[:, feature_sort]
            self.X = self.X[:, :n_components]
        elif method == "FS":
            self.fs = SelectKBest(score_func=chi2, k=self.n_components)
            self.fs.fit(self.X[: self.handicap, :], self.y[: self.handicap])
            self.X = self.fs.transform(self.X)
        else:
            print("Provide a method!")
            exit()

        self.chunk_size = chunk_size
        self.chunk_id = -1
        self.chunk = None
        self.classes_ = np.unique(self.y)
        self.X = self.X[self.handicap :]
        self.y = self.y[self.handicap :]
        if n_chunks is not None:
            self.n_chunks = n_chunks
        else:
            self.n_chunks = self.y.shape[0] // self.chunk_size
        print(self.n_chunks)

    def is_dry(self):
        return self.chunk_id >= (self.n_chunks - 1)

    def get_chunk(self):
        self.chunk_id += 1
        print("CHUNK %i" % self.chunk_id)
        self.previous_chunk = self.chunk

        start = self.chunk_id * self.chunk_size
        end = (self.chunk_id + 1) * self.chunk_size
        self.chunk = (self.X[start:end, :], self.y[start:end])

        return self.chunk


used_features = [2, 10, 50, 100, 200, 500, 1000]

for n_components in used_features:
    # PCA
    plt.figure()
    stream = StreamFromFile("data/cv.npz", n_components=n_components, method="PCA")
    clfs = [
        sl.ensembles.SEA(GaussianNB(), n_estimators=5),
        sl.ensembles.SEA(MLPClassifier(random_state=1410), n_estimators=5),
        sl.ensembles.SEA(HoeffdingTree(), n_estimators=5),
    ]
    eval = sl.evaluators.TestThenTrain(metrics=(accuracy_score))
    eval.process(stream, clfs)

    print(eval.scores, eval.scores.shape)
    print(np.mean(eval.scores, axis=1))

    # plt.plot(np.squeeze(eval.scores).T)

    for value, label, mean in zip(
        np.squeeze(eval.scores), methods, np.mean(eval.scores, axis=1)
    ):
        label += "\n{0:.3f}".format(mean[0])
        plt.plot(value, label=label)

    plt.legend(
        loc=8,
        # bbox_to_anchor=(0.5, -0.1),
        fancybox=False,
        shadow=True,
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    plt.ylim(0, 1)
    plt.title("PCA - %i" % n_components)

    plt.savefig("figures/PCA_%i" % (n_components))
    plt.savefig("foo")

    np.save("results/PCA_%i" % (n_components), eval.scores)

    # CV
    plt.figure()
    stream = StreamFromFile("data/cv.npz", n_components=n_components, method="CV")
    clfs = [
        sl.ensembles.SEA(GaussianNB(), n_estimators=5),
        sl.ensembles.SEA(MLPClassifier(random_state=1410), n_estimators=5),
        sl.ensembles.SEA(HoeffdingTree(), n_estimators=5),
    ]
    eval = sl.evaluators.TestThenTrain(metrics=(accuracy_score))
    eval.process(stream, clfs)

    print(eval.scores, eval.scores.shape)
    print(np.mean(eval.scores, axis=1))

    # plt.plot(np.squeeze(eval.scores).T)

    for value, label, mean in zip(
        np.squeeze(eval.scores), methods, np.mean(eval.scores, axis=1)
    ):
        label += "\n{0:.3f}".format(mean[0])
        plt.plot(value, label=label)

    plt.legend(
        loc=8,
        # bbox_to_anchor=(0.5, -0.1),
        fancybox=False,
        shadow=True,
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    plt.ylim(0, 1)
    plt.title("CV - %i" % n_components)

    plt.savefig("figures/CV_%i" % (n_components))
    plt.savefig("bar")

    np.save("results/CV_%i" % (n_components), eval.scores)
    plt.clf()

    # Feature selection
    plt.figure()
    stream = StreamFromFile("data/cv.npz", n_components=n_components, method="FS")
    clfs = [
        sl.ensembles.SEA(GaussianNB(), n_estimators=5),
        sl.ensembles.SEA(MLPClassifier(random_state=1410), n_estimators=5),
        sl.ensembles.SEA(HoeffdingTree(), n_estimators=5),
    ]
    eval = sl.evaluators.TestThenTrain(metrics=(accuracy_score))
    eval.process(stream, clfs)

    print(eval.scores, eval.scores.shape)
    print(np.mean(eval.scores, axis=1))

    # plt.plot(np.squeeze(eval.scores).T)

    for value, label, mean in zip(
        np.squeeze(eval.scores), methods, np.mean(eval.scores, axis=1)
    ):
        label += "\n{0:.3f}".format(mean[0])
        plt.plot(value, label=label)

    plt.legend(
        loc=8,
        # bbox_to_anchor=(0.5, -0.1),
        fancybox=False,
        shadow=True,
        ncol=5,
        fontsize=8,
        frameon=False,
    )

    plt.ylim(0, 1)
    plt.title("FS - %i" % n_components)

    plt.savefig("figures/FS_%i" % (n_components))
    plt.savefig("baz")

    np.save("results/FS_%i" % (n_components), eval.scores)
    plt.clf()
