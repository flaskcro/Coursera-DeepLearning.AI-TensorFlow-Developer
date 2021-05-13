import json
with open('./data/sarcasm.json', 'r') as f:
    datastore = json.load(f)
sentences = []
labels = []
urls = []

for item in datastore:

    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import numpy as np

train_sentences = sentences[:-5000]
test_sentences = sentences[-5000:]
train_labels = labels[:-5000]
test_labels = labels[-5000:]
training_labels_final = np.array(train_labels)
testing_labels_final = np.array(test_labels)
vocab_size = 10000
max_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token='OOV')
tokenizer.fit_on_texts(train_sentences)
train_sequences = tokenizer.texts_to_sequences(train_sentences)
test_sentences = tokenizer.texts_to_sequences((test_sentences))

train_padded = pad_sequences(train_sequences, padding='post', truncating='post', maxlen=max_length)
test_padded = pad_sequences(test_sentences, padding='post', truncating='post', maxlen=max_length)

train_padded = np.array(train_padded)
train_labels = np.array(train_labels)
test_padded = np.array(test_padded)
test_labels = np.array(test_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 16, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

import os
import shutil

log_dir = './log'
if os.path.exists(log_dir):
    shutil.rmtree(log_dir)
os.mkdir(log_dir)

from tensorflow.keras.callbacks import TensorBoard

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
num_epochs = 10
history = model.fit(train_padded, train_labels,
                    validation_data=(test_padded, test_labels),
                    epochs=num_epochs, verbose=2, callbacks=TensorBoard(log_dir=log_dir, histogram_freq=1))