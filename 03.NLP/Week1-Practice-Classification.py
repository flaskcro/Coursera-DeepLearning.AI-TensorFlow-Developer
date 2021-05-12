import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

stopwords = [ "a", "about", "above", "after", "again", "against",
             "all", "am", "an", "and", "any", "are", "as", "at", "be",
             "because", "been", "before", "being", "below", "between",
             "both", "but", "by", "could", "did", "do", "does", "doing",
             "down", "during", "each", "few", "for", "from", "further",
             "had", "has", "have", "having", "he", "he'd", "he'll",
             "he's", "her", "here", "here's", "hers", "herself", "him",
             "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm",
             "i've", "if", "in", "into", "is", "it", "it's", "its", "itself",
             "let's", "me", "more", "most", "my", "myself", "nor", "of", "on",
             "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
             "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
             "should", "so", "some", "such", "than", "that", "that's", "the",
             "their", "theirs", "them", "themselves", "then", "there", "there's",
             "these", "they", "they'd", "they'll", "they're", "they've", "this",
             "those", "through", "to", "too", "under", "until", "up", "very",
             "was", "we", "we'd", "we'll", "we're", "we've", "were", "what",
             "what's", "when", "when's", "where", "where's", "which", "while",
             "who", "who's", "whom", "why", "why's", "with", "would", "you",
             "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
             "yourselves" ]

labels = []
sentences = []
with open('./data/bbc-text.csv', 'r') as csvfile:
    rdr = csv.reader(csvfile, delimiter=',')
    next(rdr)
    for row in rdr:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)

labels = labels[:1500]
sentences = sentences[:1500]

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post', truncating='post', maxlen=100)

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
label_word_index = label_tokenizer.word_index
label_seq = label_tokenizer.texts_to_sequences(labels)

label_seq = [x[0] for x in label_seq]
label_seq = tf.keras.utils.to_categorical(label_seq)

X_train, X_test, y_train, y_test = train_test_split(padded, label_seq, train_size=0.8, random_state=72)
print(len(word_index), padded.shape, y_train.shape, y_test.shape)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Embedding(len(word_index), padded.shape[1])) #len(word_index), len(padded)
model.add(tf.keras.layers.LSTM(padded.shape[1], activation='tanh'))
model.add(tf.keras.layers.Dense(y_train.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=100, epochs=20)

print('Accuracy :', model.evaluate(X_test, y_test)[1])