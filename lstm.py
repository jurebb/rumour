import json
from pomocni import tree2branches
import numpy as np

from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Sequential
from keras.layers import Activation
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
from keras.layers import Bidirectional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_set import load_dataset, load_true_labels

# %%
DIR = 'data\\data\\rumoureval-2019-training-data\\'


def class_to_onehot(y):
    y_oh = np.zeros((len(y), max(y) + 1))
    y_oh[range(len(y)), y] = 1
    return y_oh


def main():

    data = load_dataset()
    counter = 0
    data2 = []
    data3 = []

    for base_text in data['train']:
        counter += 1
        for reply in base_text['replies']:
            counter += 1
            data2.append(reply)
            data3.append(base_text['source']['text'])

    df = np.asarray(data2)
    df2 = np.asarray(data3)
    x_text = []
    y = []
    for d in df:
        x_text.append(d['text'])
        y.append(d['label'])

    x_text = np.asarray(x_text)
    y = np.asarray(y)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    y = le.fit_transform(y)

    x_train_str, x_test_str, y_train, y_test = train_test_split(x_text, y, test_size=0.2)
    
    GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
    EMBEDDING_DIM = 200
    MAX_SEQUENCE_LENGTH = 50
    MAX_NB_WORDS = 200000

    texts = x_train_str

    tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    x_train = data
    y_train = class_to_onehot(y_train)

    sequences_test = tokenizer.texts_to_sequences(x_test_str)
    x_test = pad_sequences(sequences_test, maxlen=MAX_SEQUENCE_LENGTH)
    #y_test = class_to_onehot(y_test)

    embeddings_index = {}
    f = open(GLOVE_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            pass
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print(embedding_matrix.shape)

    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    model = Sequential()

    embedding_layer = Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)

    model.add(embedding_layer)

    model.add(Bidirectional(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))) # probao ali nije davalo znacajne razlike a trenira se nesto duze
    #model.add(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2))
    
    model.add(Activation('sigmoid'))

    model.add(Dense(4))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit(x_train, y_train, nb_epoch=10, batch_size=64)  # nb_epoch=50
    
    preds_test = model.predict(x_test, 100)
    preds_test = [np.argmax(x) for x in preds_test]
    print(preds_test)
    print(y_test)
    print(accuracy_score(preds_test, y_test))

main()