import pandas as pd

import json
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

from sequential.prepare_seq_data import *
from sequential.additional_features import *
from sequential.additional_computed_features import *

from numpy.random import seed
from tensorflow import set_random_seed
seed(12)
set_random_seed(22)

_DATA_DIR = "C:\\Users\\viktor\\Projects\\Python\\data_set\\data"
MAX_BRANCH_LENGTH = 24
NUMBER_OF_CLASSES = 4
# GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
GLOVE_DIR = '/home/interferon/Documents/dipl_projekt/glove/glove.twitter.27B.200d.txt'

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
    for i in range(len(y_test)):
        if x_test_text_branchifyed[i] in x_test_text_original: 
            y_test2.append(y_test[i])
            preds_test2.append(preds_test[i])
            x_test_text_original.remove(x_test_text_branchifyed[i])

    return preds_test2, y_test2


def concat_features(feature_functions_list, train_set, test_set, dev_set, train_y, test_y, dev_y):
    new_train_set = []
    new_test_set = None
    new_dev_set = None
    if test_set is not None:
        new_test_set = []
    if dev_set is not None:
        new_dev_set = []

    #TODO wont currently work for multiple function features, it will append to train_set all the time

    for feature_function in feature_functions_list:
        tr_x_feature_val, ts_x_feature_val, dv_x_feature_val = feature_function()

        tweet_counter = 0
        for i in range(len(train_set)):
            new_branch = []
            for j in range(len(train_set[i])):
                if not np.array_equal(train_y[i][j], np.zeros(NUMBER_OF_CLASSES)):
                    # train_set[i][j] = np.append(train_set[i][j], tr_x_feature_val[tweet_counter])

                    try:
                        new_branch.append(np.concatenate((train_set[i][j], np.array([tr_x_feature_val[tweet_counter]]))))
                    except ValueError:
                        new_branch.append(np.concatenate((train_set[i][j], np.array(tr_x_feature_val[tweet_counter])))) #WITHOUT EXTRA []
                    tweet_counter += 1
                else:
                    try:
                        new_branch.append(
                            np.concatenate((train_set[i][j], np.array([0] * len(tr_x_feature_val[0])))))
                    except TypeError:
                        new_branch.append(
                            np.concatenate((train_set[i][j], np.array([0]))))

            new_train_set.append(new_branch)
        new_train_set = np.asarray(new_train_set)

        if test_set is not None:
            tweet_counter = 0
            for i in range(len(test_set)):
                new_branch = []
                for j in range(len(test_set[i])):
                    if not np.array_equal(test_y[i][j], np.zeros(NUMBER_OF_CLASSES)):
                        # train_set[i][j] = np.append(train_set[i][j], tr_x_feature_val[tweet_counter])
                        try:
                            new_branch.append(
                                np.concatenate((test_set[i][j], np.array([ts_x_feature_val[tweet_counter]]))))
                        except ValueError:
                            new_branch.append(
                                np.concatenate((test_set[i][j], np.array(ts_x_feature_val[tweet_counter]))))
                        tweet_counter += 1
                    else:
                        try:
                            new_branch.append(
                                np.concatenate((test_set[i][j], np.array([0] * len(ts_x_feature_val[0])))))
                        except TypeError:
                            new_branch.append(
                                np.concatenate((test_set[i][j], np.array([0]))))
                new_test_set.append(new_branch)
            new_test_set = np.asarray(new_test_set)

        if dev_set is not None:
            tweet_counter = 0
            for i in range(len(dev_set)):
                new_branch = []
                for j in range(len(dev_set[i])):
                    if not np.array_equal(dev_y[i][j], np.zeros(NUMBER_OF_CLASSES)):
                        # train_set[i][j] = np.append(train_set[i][j], tr_x_feature_val[tweet_counter])
                        try:
                            new_branch.append(
                                np.concatenate((dev_set[i][j], np.array([dv_x_feature_val[tweet_counter]]))))
                        except ValueError:
                            new_branch.append(
                                np.concatenate((dev_set[i][j], np.array(dv_x_feature_val[tweet_counter]))))
                        tweet_counter += 1
                    else:
                        try:
                            new_branch.append(
                                np.concatenate((dev_set[i][j], np.array([0] * len(dv_x_feature_val[0])))))
                        except TypeError:
                            new_branch.append(
                                np.concatenate((dev_set[i][j], np.array([0]))))
                new_dev_set.append(new_branch)
            new_dev_set = np.asarray(new_dev_set)

        return new_train_set, new_test_set, new_dev_set


def main():
    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    MAX_BRANCH_LENGTH = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH=', MAX_BRANCH_LENGTH)

    embeddings_index = make_embeddings_index()

    x_train_temp = transform_data(tr_x, embeddings_index) 
    y_train_temp = transform_labels(tr_y) 

    x_test_temp = transform_data(dv_x, embeddings_index) 
    y_test_temp = transform_labels(dv_y)

    twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    #### feature engineering
    # for new_feature_f in [twitter_user_description_feature, twitter_user_id_feature, twitter_retweet_count_feature, twitter_profile_favourites_count, twitter_profile_use_background_image_feature, twitter_time_feature]:
    for new_feature_f in [twitter_user_mention_count_feature]:
        x_train, _, x_test = concat_features([new_feature_f], x_train_temp, None, x_test_temp, y_train_temp, None, y_test_temp)
        y_train = y_train_temp
        y_test = y_test_temp
        print('x_train.shape', x_train.shape)

    ####
        print('x_train.shape', x_train.shape)

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

        preds_test = model.predict(x_test, 100) 
        preds_test = [np.argmax(xx) for x in preds_test for xx in x] #includes predictions for padded data
        y_test = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test for xx in x] #predictions for padded data added as str None
        print('len(y_test)', len(y_test))
        print('accuracy_score(preds_test, y_test)', accuracy_score(preds_test, y_test))

        preds_test, y_test = remove_padded_data(preds_test, y_test)
        preds_test, y_test = remove_duplicated_data(preds_test, y_test, d_tw, dv_x)

        print('accuracy_score(preds_test, y_test) after removing padded/duplicated', accuracy_score(preds_test, y_test))
        print('preds_test', preds_test)

main()
