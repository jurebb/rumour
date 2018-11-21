import pandas as pd

import json
from pomocni import tree2branches
import numpy as np

from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
from keras.layers import Bidirectional

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def main():   
    _DATA_DIR = "data" 
    NUMBER_OF_CLASSES = 4

    os.chdir(_DATA_DIR)
    x_train = np.load('x_train.npy')
    x_test = np.load('x_test.npy')
    y_train = np.load('y_train.npy')
    y_test = np.load('y_test.npy')

    model = Sequential()
    #model.add(Bidirectional(LSTM(units=100, dropout=0.2, recurrent_dropout=0.2, input_shape=(x_train.shape[1], x_train.shape[2]))))
    model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))

    model.add(Activation('sigmoid'))
    model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES + 1)))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, y_train, nb_epoch=8, batch_size=64)  # nb_epoch=50

    preds_train = model.predict(x_train, 100)
    print(preds_train[0])
    preds_train = [np.argmax(xx) for x in preds_train for xx in x]
    y_train = [np.argmax(xx) for x in y_train for xx in x]

    print('train score')
    #print(preds_train)
    #print(y_train)
    print(accuracy_score(preds_train, y_train))

    preds_test = model.predict(x_test, 100)
    preds_test = [np.argmax(xx) for x in preds_test for xx in x]
    y_test = [np.argmax(xx) for x in y_test for xx in x]
    print(preds_test)
    print(y_test)
    print(accuracy_score(preds_test, y_test))

    y_test2 = []
    preds_test2 = []
    for i in range(len(y_test)):
        if y_test[i] != NUMBER_OF_CLASSES:
            y_test2.append(y_test[i])
            preds_test2.append(preds_test[i])

    print(accuracy_score(preds_test2, y_test2))
    

main()