import numpy as np
data = np.load("data/cv.npz")
X_, y = data["X"], data["y"]

print(X_.shape, y.shape)
