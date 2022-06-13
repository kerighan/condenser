import numpy as np
# from convectors.classifier.layers import WeightedAttention
from convectors.layers import Lemmatize, Sequence, Tokenize
from keras_self_attention import SeqSelfAttention
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras.layers import Dense, Dropout, Embedding, Input
from tensorflow.keras.models import Model

from condenser import Condenser, WeightedAttention

np.random.seed(0)

newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

MAX_FEATURES = 30000
EMBEDDING_DIM = 200
MAXLEN = 200

# use convectors as a preprocessing pipeline
nlp = Tokenize()
nlp += Lemmatize(lang="en")
nlp += Sequence(maxlen=MAXLEN, max_features=MAX_FEATURES)

# process train data
X_train = nlp(newsgroups_train.data)
y_train = newsgroups_train.target
# process test data
X_test = nlp(newsgroups_test.data)
y_test = newsgroups_test.target

# get number of features
n_features = nlp["Sequence"].n_features + 1
print(f"n_features={n_features}")

# build model
inp = Input(shape=(MAXLEN,))
x = Embedding(n_features, EMBEDDING_DIM, mask_zero=True)(inp)
x = SeqSelfAttention(attention_width=10)(x)
x = SeqSelfAttention(attention_width=10)(x)
# x = Condenser(
#     attention_dim=1,
#     n_sample_points=25,
#     use_residual=True,
#     use_reducer=True,
#     theta_trainable=True,
#     attention_activation="tanh",
#     reduce_trainable=False,
#     reduce_dim=128)(x)
att = WeightedAttention(hidden_dim=24)
x = att(x)
x = Dropout(.2)(x)
x = Dense(64, activation="tanh")(x)
out = Dense(20, activation="softmax")(x)

model = Model(inp, out)
model.compile("nadam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train,
          batch_size=200, epochs=1,
          validation_data=(X_test, y_test),
          shuffle=True)
print(att.weights[-1])
print(att.weights[-2])
