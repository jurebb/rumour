import pandas as pd

import json
import numpy as np

import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import os
from itertools import combinations

from sequential.prepare_seq_data import *

# GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
GLOVE_DIR = '/home/interferon/Documents/dipl_projekt/glove/glove.twitter.27B.200d.txt'
NUMBER_OF_CLASSES = 4

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


def transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH):
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



def transform_labels(tr_y, MAX_BRANCH_LENGTH):
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