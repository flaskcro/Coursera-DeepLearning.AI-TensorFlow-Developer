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

tokenizer = Tokenizer(oov_token='OOV')
tokenizer.fit_on_texts(sentences)
sequnces = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequnces, padding='post')
print(sentences[0])
print(sequnces[0])
print(padded[0])
print(padded.shape)