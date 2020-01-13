# https://www.kaggle.com/kernels/svzip/6878570
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os

fake_news = pd.read_csv("input/fake.csv")
real_news = pd.read_csv("input/real_news.csv")

print(fake_news.shape)
print(real_news.shape)

print(list(fake_news.columns))
print(list(real_news.columns))


real_news2 = real_news[["title", "content", "publication"]]
real_news2["label"] = "real"
print(real_news2.head(10))

fake_news2 = fake_news[["title", "text", "site_url"]]
fake_news2["label"] = "fake"
print(fake_news2.head(10))

# let's obtain all the unique site_urls
site_urls = fake_news2["site_url"]

# let's remove the domain extensions
site_urls2 = [x.split(".", 1)[0] for x in site_urls]

# now let's replace the old site_url column
fake_news2["site_url"] = site_urls2

print(fake_news2.head())

# let's rename the features in our datasets to be the same
newlabels = ["title", "content", "publication", "label"]
real_news2.columns = newlabels
fake_news2.columns = newlabels

# let's concatenate the dataframes
frames = [fake_news2, real_news2]
news_dataset = pd.concat(frames)
print(news_dataset)

# news_dataset.to_csv("input/kaggle_news_dataset.csv", encoding="utf-8")
