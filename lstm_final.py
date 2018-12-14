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
from sklearn.feature_extraction.text import TfidfVectorizer

from data_set import load_dataset, load_true_labels
from sequential.prepare_seq_data import *

_DATA_DIR = "data"

def class_to_onehot(y, max_y = None):
    if max_y == None:
        max_y = max(y)
    y_oh = np.zeros((len(y), max_y + 1))
    y_oh[range(len(y)), y] = 1
    return y_oh

def make_embeddings_index():
    embeddings_index = {}
    f = open(GLOVE_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
            if coefs[1] < -113:
                print(coefs)
        except:
            pass
    f.close()
    return embeddings_index

def transform_data(tr_x, embeddings_index):
    x = []
    for podatak in tr_x:
        temp_podatak = []
        for sentence in podatak:
            suma = []
            br = 0
            for word in sentence.split():
                word = word.lower()
                if len(word) > 1 and word[-1] in '.!?*,':
                    word = word[:-1]
                if word in embeddings_index:
                    if len(suma) == 0:
                        for broj in embeddings_index[word]:
                            suma.append(broj)
                        suma = np.asarray(suma)
                    else:
                        suma += embeddings_index[word]
                    br += 1
            if len(suma) == 0:
                for broj in embeddings_index['.']:
                    suma.append(broj)
                suma = np.asarray(suma)
                br += 1
            temp_podatak.append(suma / br)

        temp_podatak = np.asarray(temp_podatak)
        if temp_podatak.shape[0] < MAX_BRANCH_LENGTH:
            temp_podatak = np.concatenate((temp_podatak, np.zeros((MAX_BRANCH_LENGTH - temp_podatak.shape[0], temp_podatak.shape[1]))), axis = 0)
        else:
            temp_podatak = temp_podatak[:MAX_BRANCH_LENGTH]
        x.append(temp_podatak)
    x = np.asarray(x)
    return x

def transform_labels(tr_y):
    from sklearn.externals import joblib
    le = joblib.load('le.pkl')
    y_train = []
    for yy in tr_y:
        yy = le.transform(yy)
        temp = class_to_onehot(yy, NUMBER_OF_CLASSES - 1)

        if temp.shape[0] < MAX_BRANCH_LENGTH:
            to_add = np.zeros((MAX_BRANCH_LENGTH - temp.shape[0], temp.shape[1]))
            temp = np.concatenate((temp, to_add), axis = 0)
        else:
            temp = temp[:MAX_BRANCH_LENGTH]

        y_train.append(temp)
    y_train = np.asarray(y_train)
    return y_train

def remove_padded_data(preds_test, y_test):
    y_test2 = []
    preds_test2 = []
    for i in range(len(y_test)):
        if y_test[i] != 'None':
            y_test2.append(y_test[i])
            preds_test2.append(preds_test[i])

    return preds_test2, y_test2

def remove_duplicated_data(preds_test, y_test, d_tw, dv_x):
    x_test_text_branchifyed = []
    for podatak in dv_x:
        for sentence in podatak:
            x_test_text_branchifyed.append(sentence)

    dev = d_tw['dev']
    
    x_test_text_original = []
    for nesto in dev:
        for tekst in nesto['replies']:
            x_test_text_original.append(tekst['text'])

        x_test_text_original.append(nesto['source']['text'])

    y_test2 = []
    preds_test2 = []
    index_of_text = dict()
    for i in range(len(y_test)):
        if x_test_text_branchifyed[i] in x_test_text_original: 
            y_test2.append(y_test[i])
            preds_test2.append(preds_test[i])
            x_test_text_original.remove(x_test_text_branchifyed[i])
            index_of_text[x_test_text_branchifyed[i]] = i
        else:
            j = index_of_text[x_test_text_branchifyed[i]]
            if preds_test[i] == preds_test[j]:
                print('jednaki')
                print(i)
                print(j)
                print(preds_test[i])
                print(preds_test[j])
            else:
                print('razliciti')
                print(preds_test[i])
                print(preds_test[j])
                

    return preds_test2, y_test2


MAX_BRANCH_LENGTH = 25
NUMBER_OF_CLASSES = 4
GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'

def main():
    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index)
    y_train = transform_labels(tr_y)

    model = Sequential()
    model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Activation('sigmoid'))
    model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, y_train, nb_epoch=8, batch_size=64)  # nb_epoch=50

    #preds_train = model.predict(x_train, 100)
    #preds_train = [np.argmax(xx) for x in preds_train for xx in x]
    #y_train = [np.argmax(xx) for x in y_train for xx in x]

    x_test = transform_data(dv_x, embeddings_index)
    y_test = transform_labels(dv_y)

    preds_test = model.predict(x_test, 100) 
    preds_test = [np.argmax(xx) for x in preds_test for xx in x] #includes predictions for padded data
    y_test = [np.argmax(xx)  if np.max(xx) != 0 else 'None' for x in y_test for xx in x] #predictions for padded data added as str None
    print(len(y_test))
    print(accuracy_score(preds_test, y_test))

    preds_test, y_test = remove_padded_data(preds_test, y_test)
    preds_test, y_test = remove_duplicated_data(preds_test, y_test, d_tw, dv_x)

    print(accuracy_score(preds_test, y_test))
    print(preds_test)

main()

    
    