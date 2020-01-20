"""
Extracting features using count vectorizer
"""
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

n_features = 2000
crop = 0

#
# Get data and remove NaN's
#
# ID, title, body, source, label
print("Get data")
dataset = pd.read_csv("input/kaggle_news_dataset.csv")
nanremover = pd.isnull(dataset).values
nanremover = np.sum(nanremover, axis=1) == 0

data = dataset.values[nanremover]
if crop > 0:
    data = data[:crop]

titles = data[:, 1]
contents = data[:, 2]
labels = data[:, -1]

#
# Extract features by CV
#
print("Extract features by Count Vectorizer")
vectorizer = CountVectorizer(stop_words="english", max_features=n_features)

X_titles_cv = vectorizer.fit_transform(titles)
X_contents_cv = vectorizer.fit_transform(contents)

X_titles = X_titles_cv.toarray()
X_contents = X_contents_cv.toarray()

print(np.max(X_titles, axis=0), X_titles.shape)
print(np.max(X_contents, axis=0), X_contents.shape)

#
# Prepare dataset
#
X = np.zeros((data.shape[0], 2 * n_features)).astype("int")
X[:, :n_features] = X_titles
X[:, n_features:] = X_contents
y = (labels == "fake").astype(int)

np.savez("data/cv", X=X, y=y)
