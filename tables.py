import numpy as np

used_features = [2, 10, 50, 100, 200, 500, 1000]
methods = ["GNB", "MLP", "HT"]

print("SEA\n")
for n_components in used_features:
    sea_pca = np.load('results/SEA-PCA_%i.npy' % (n_components))
    sea_cvf = np.load('results/SEA-CV_%i.npy' % (n_components))
    sea_fs = np.load('results/SEA-FS_%i.npy' % (n_components))

    mean_sea_pca = np.mean(sea_pca, axis=1)
    mean_sea_cvf = np.mean(sea_cvf, axis=1)
    mean_sea_fs = np.mean(sea_fs, axis=1)

    sea = ["%.3f" % val for val in mean_sea_pca] + ["%.3f" % val for val in mean_sea_cvf] + ["%.3f" % val for val in mean_sea_fs]

    print(("%i & " % n_components) + " & ".join(sea) + " \\\\")
print("\n\n")

print("OB\n")
for n_components in used_features:
    ob_pca = np.load('results/OB-PCA_%i.npy' % (n_components))
    ob_cvf = np.load('results/OB-CV_%i.npy' % (n_components))
    ob_fs = np.load('results/OB-FS_%i.npy' % (n_components))

    mean_ob_pca = np.mean(ob_pca, axis=1)
    mean_ob_cvf = np.mean(ob_cvf, axis=1)
    mean_ob_fs = np.mean(ob_fs, axis=1)

    ob = ["%.3f" % val for val in mean_ob_pca] + ["%.3f" % val for val in mean_ob_cvf] + ["%.3f" % val for val in mean_ob_fs]

    print(("%i & " % n_components) + " & ".join(ob) + " \\\\")
print("\n\n")

print("SIN\n")
for n_components in used_features:
    sin_pca = np.load('results/SIN-PCA_%i.npy' % (n_components))
    sin_cvf = np.load('results/SIN-CV_%i.npy' % (n_components))
    sin_fs = np.load('results/SIN-FS_%i.npy' % (n_components))

    mean_sin_pca = np.mean(sin_pca, axis=1)
    mean_sin_cvf = np.mean(sin_cvf, axis=1)
    mean_sin_fs = np.mean(sin_fs, axis=1)

    sin = ["%.3f" % val for val in mean_sin_pca] + ["%.3f" % val for val in mean_sin_cvf] + ["%.3f" % val for val in mean_sin_fs]

    print(("%i & " % n_components) + " & ".join(sin) + " \\\\")
print("\n\n")
