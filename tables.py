import numpy as np



used_features = [2, 10, 50, 100, 200, 500, 1000]
methods = ["GNB", "KNN", "MLP", "SVC", "CART"]

for n_components in used_features:
    pca = np.load('results/PCA_%i.npy' % (n_components))
    cvf = np.load('results/CV_%i.npy' % (n_components))

    mean_pca = np.mean(pca, axis=1)
    mean_cvf = np.mean(cvf, axis=1)

    a = ["%.3f" % val for val in mean_pca] + ["%.3f" % val for val in mean_cvf]
    print(("%i & " % n_components) + " & ".join(a) + " \\\\")
