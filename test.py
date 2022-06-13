import pandas as pd
from convectors.layers import Lemmatize, Sequence, Sub, TfIdf, Tokenize
from sklearn.datasets import fetch_20newsgroups
from stc import SparseTensorClassifier

from condenser import Condenser

train = fetch_20newsgroups(subset='train')
test = fetch_20newsgroups(subset='test')

MAX_FEATURES = 30000
EMBEDDING_DIM = 96
MAXLEN = 96

# use convectors as a preprocessing pipeline
nlp = Tokenize()
nlp += TfIdf(min_df=20)
X = nlp(train.data).todense()
print(X.shape)

df = pd.DataFrame(X)
columns = list(df.columns)
df["class_type"] = list(train.target)
print(df)
STC = SparseTensorClassifier(targets="class_type", features=columns)
STC.fit(df)

X = nlp(test.data).todense()
df = pd.DataFrame(X)
columns = list(df.columns)
df["class_type"] = list(test.target)
print(df)
labels, probability, explainability = STC.predict(df)
print(labels)
