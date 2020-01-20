import numpy as np
import strlearn as sl
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt


class StreamFromFile:
    def __init__(self, filename, chunk_size=250):
        self.filename = filename
        self.data = np.load("data/cv.npz")
        self.X, self.y = self.data["X"], self.data["y"]
        self.X, self.y = shuffle(self.X, self.y, random_state=0)
        self.chunk_size = chunk_size
        self.chunk_id = 0
        self.chunk = None
        self.classes_ = np.unique(self.y)
        self.n_chunks = self.y.shape[0] // self.chunk_size
        print(self.n_chunks)

    def is_dry(self):
        return self.chunk_id >= (self.n_chunks - 1)

    def get_chunk(self):
        self.previous_chunk = self.chunk

        start = self.chunk_id * self.chunk_size
        end = (self.chunk_id + 1) * self.chunk_size
        self.chunk = (self.X[start:end, :], self.y[start:end])

        self.chunk_id += 1
        return self.chunk


stream = StreamFromFile("data/cv.npz")
clfs = [GaussianNB(), MLPClassifier()]
eval = sl.evaluators.TestThenTrain(metrics=(accuracy_score))
eval.process(stream, clfs)

print(eval.scores, eval.scores.shape)

plt.plot(np.squeeze(eval.scores).T)
plt.ylim(0, 1)

plt.savefig("foo.png")


# for i in range(10):
#    X, y = stream.get_chunk()
#    print(y, np.unique(y, return_counts=True))
