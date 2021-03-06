from convectors.layers import Lemmatize, Sequence, Tokenize
from keras_self_attention import SeqSelfAttention
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense, Embedding, Input
from tensorflow.keras.models import Model

from condenser import Condenser

MAX_FEATURES = 200000
EMBEDDING_DIM = 400
MAXLEN = 600

# get training data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# use convectors as a preprocessing pipeline
nlp = Tokenize()
nlp += Sequence(maxlen=MAXLEN, max_features=MAX_FEATURES)

# process train data
X_train = nlp(newsgroups_train.data)
y_train = newsgroups_train.target
# process test data
X_test = nlp(newsgroups_test.data)
y_test = newsgroups_test.target

# get number of features
n_features = nlp["Sequence"].n_features + 1

# build model
inp = Input(shape=(MAXLEN,))
x = Embedding(n_features, EMBEDDING_DIM, mask_zero=True)(inp)
x = SeqSelfAttention(units=64,
                     attention_width=10,
                     attention_activation="tanh",
                     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(x)
x = SeqSelfAttention(units=64,
                     attention_width=10,
                     attention_activation="tanh",
                     attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(x)
x = Condenser(n_sample_points=15, reducer_dim=96, theta_regularizer=None,
              attention_activation="relu")(x)
x = Dense(48, activation="tanh")(x)
out = Dense(20, activation="softmax")(x)

# create and fit model
model = Model(inp, out)
model.compile("nadam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train,
          batch_size=20, epochs=5,
          validation_data=(X_test, y_test),
          shuffle=True)
# >>> val_accuracy=0.8716
