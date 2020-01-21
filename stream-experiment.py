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

BASE_CV = 1000


class StreamFromFile:
    def __init__(
        self, filename, chunk_size=250, n_components=4, n_chunks=None, use_PCA=True
    ):
        self.filename = filename
        self.data = np.load("data/cv.npz")
        self.X, self.y = self.data["X"], self.data["y"]
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        self.n_components = n_components

        if use_PCA:
            self.pca = PCA(self.n_components).fit(self.X)
            self.X = self.pca.transform(self.X)

        else:
            wordsum = np.sum(self.X, axis=0)
            feature_sort = np.argsort(-wordsum)

            self.X = self.X[:, feature_sort]
            self.X = self.X[:, :n_components]

        self.chunk_size = chunk_size
        self.chunk_id = -1
        self.chunk = None
        self.classes_ = np.unique(self.y)
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


used_features = [2, 10, 50, 100, 200, 500]

for n_components in used_features:
    # PCA
    stream = StreamFromFile("data/cv.npz", n_components=10, n_chunks=10, use_PCA=True)
    clfs = [
        sl.ensembles.SEA(GaussianNB(), n_estimators=5),
        sl.ensembles.SEA(KNeighborsClassifier(), n_estimators=5),
        sl.ensembles.SEA(MLPClassifier(), n_estimators=5),
        sl.ensembles.SEA(SVC(probability=True), n_estimators=5),
        sl.ensembles.SEA(DecisionTreeClassifier(), n_estimators=5),
    ]
    eval = sl.evaluators.TestThenTrain(metrics=(accuracy_score))
    eval.process(stream, clfs)

    print(eval.scores, eval.scores.shape)
    print(np.mean(eval.scores, axis=1))

    plt.plot(np.squeeze(eval.scores).T)
    plt.ylim(0, 1)
    plt.title("PCA - %i" % n_components)

    plt.savefig("figures/PCA_%i" % (n_components))
    plt.savefig("foo")

    np.save("results/PCA_%i" % (n_components), eval.scores)

    plt.clf()

    # CV
    stream = StreamFromFile("data/cv.npz", n_components=10, n_chunks=3, use_PCA=False)
    clfs = [
        sl.ensembles.SEA(GaussianNB(), n_estimators=5),
        sl.ensembles.SEA(KNeighborsClassifier(), n_estimators=5),
        sl.ensembles.SEA(MLPClassifier(), n_estimators=5),
        sl.ensembles.SEA(SVC(probability=True), n_estimators=5),
        sl.ensembles.SEA(DecisionTreeClassifier(), n_estimators=5),
    ]
    eval = sl.evaluators.TestThenTrain(metrics=(accuracy_score))
    eval.process(stream, clfs)

    print(eval.scores, eval.scores.shape)
    print(np.mean(eval.scores, axis=1))

    plt.plot(np.squeeze(eval.scores).T)
    plt.ylim(0, 1)
    plt.title("CV - %i" % n_components)

    plt.savefig("figures/CV_%i" % (n_components))
    plt.savefig("bar")

    np.save("results/CV_%i" % (n_components), eval.scores)
    plt.clf()
