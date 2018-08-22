import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF
from pathlib import Path
import utils

train_path = Path.cwd().joinpath('data/semeval-2016/train.csv')
test_path = Path.cwd().joinpath('data/semeval-2016/test.csv')

# Read data
data_train = pd.read_csv(train_path)
data_test = pd.read_csv(test_path)

# Save all words and tags as a list
max_len = 75
words = list(set(data_train['Word'].values))
tags = list(set(data_train["Tag"].values))

word2idx = {w: i + 1 for i, w in enumerate(words)}
n_words = len(word2idx)

tag2idx = {t: i + 1 for i, t in enumerate(tags)}
tag2idx['<pad>'] = 0
n_tags = len(tag2idx) # Due to <pad>, here total tag number is from 17 to 18

# Convert data to sentences
data = data_train
getter = utils.SentenceGetter(data)
sentences = getter.sentences # get all sentences

# Word2inx & Padding for X
X = [[word2idx[w[0]] for w in s] for s in sentences]
X_train = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

# Word2inx & Padding for y
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)

# Get one-hot labels
y_train = [to_categorical(i, num_classes=n_tags) for i in y]


#==============Bi-LSTM CRF=============
input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="tanh"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)
model.compile(optimizer="adam", loss=crf.loss_function, metrics=[crf.accuracy])
model.summary()

history = model.fit(X_train, np.array(y_train), batch_size=32, epochs=8,
                    validation_split=0.1, verbose=1)

# Predict on test dataset
data = data_test

getter = utils.SentenceGetter(data)
sentences = getter.sentences  # get all sentences

# Word2inx & Padding for X
X = [[word2idx.get(w[0], 0) for w in s] for s in sentences]
X_test = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)

# Word2inx & Padding for y
y = [[tag2idx[w[1]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=0)

# Get one-hot labels
y_test = [to_categorical(i, num_classes=n_tags) for i in y]

# Predictions.
idx2word = {value: key for key, value in word2idx.items()}
idx2tag = {value: key for key, value in tag2idx.items()}

true_all = np.argmax(y_test, -1)
true_all_tags = [[idx2tag[idx] for idx in s if idx!=0] for s in true_all]

p_all = model.predict(np.array(X_test)) # (4796, 75, 18)
p_all= np.argmax(p_all, axis=-1) # (4796, 75)
p_all_tags = [[idx2tag[idx] for idx in s] for s in p_all] # ['B-gpe', 'O', 'O', 'O']

for i, true in enumerate(true_all_tags):  # align the length with ture_all_tags
    length = len(true)
    p_all_tags[i] = p_all_tags[i][:length]

p_all_tags = [[x.replace('<pad>', 'O') for x in s] for s in p_all_tags] # replace '<pad>' with 'O'

# Output one example
print("{:15}||{:5}||{}".format("Word", "True", "Pred"))
print(30 * "=")
for w, t, pred in zip(sentences[3], true_all_tags[3], p_all_tags[3]):
    if w != 0:
        print("{:15}: {:5} {}".format(w[0], w[1], pred))

# Evaluation
from seqeval.metrics import f1_score, classification_report
print(f1_score(true_all_tags, p_all_tags))
print(classification_report(true_all_tags, p_all_tags))
