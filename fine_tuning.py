import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from keras_self_attention import SeqSelfAttention
from sklearn.datasets import fetch_20newsgroups
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense,
                                     Embedding, Input, LayerNormalization)
from tensorflow.keras.models import Model

from condenser import Condenser, WeightedAttention

MAX_FEATURES = 200000
EMBEDDING_DIM = 400
MAXLEN = 600

# get training data
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')

# process train data
X_train = tf.constant(newsgroups_train.data)
y_train = newsgroups_train.target
# process test data
X_test = tf.constant(newsgroups_test.data)
y_test = newsgroups_test.target

# build model
text_input = Input(shape=(), dtype=tf.string)
preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder_inputs = preprocessor(text_input)
mask = encoder_inputs["input_mask"]
encoder = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2",
    trainable=True)
outputs = encoder(encoder_inputs)
sequence_output = outputs["sequence_output"]
pooled_output = outputs["pooled_output"]

# sequence_output = Dense(128, activation="tanh")(sequence_output)
# sequence_output = LayerNormalization()(sequence_output)

# 0.7779
# x = BatchNormalization()(sequence_output)
x = Condenser(n_sample_points=15,
              sampling_bounds=(1e-6, 1),
              reducer_dim=128,
              use_residual=False,
              attention_activation="relu")(sequence_output, mask=mask)

# x = WeightedAttention(32)(sequence_output, mask=mask)  # 0.7712
x = Dense(48, activation="tanh")(x)
out = Dense(20, activation="softmax")(x)

# create and fit model
model = Model(text_input, out)
model.compile("nadam", "sparse_categorical_crossentropy", metrics=["accuracy"])
model.summary()
model.fit(X_train, y_train,
          batch_size=20, epochs=5,
          validation_data=(X_test, y_test),
          shuffle=True)
# >>> val_accuracy=0.8716
