import pandas as pd

import json
import numpy as np

import keras
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
from itertools import combinations

from sequential.prepare_seq_data import *
from sequential.additional_features import *
from sequential.additional_features_reddit import *
from sequential.additional_computed_features import *
from sequential.additional_computed_features_reddit import *
from sequential.feature_utils import *
from task_b.prepare_data import *
from task_b.sdqc_model import *
from sklearn.externals import joblib

from numpy.random import seed
from tensorflow import set_random_seed
seed(12)
set_random_seed(22)

MAX_BRANCH_LENGTH_task_b = -1
NUMBER_OF_CLASSES_task_b = 3
GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
#  GLOVE_DIR = '/home/interferon/Documents/dipl_projekt/glove/glove.twitter.27B.200d.txt'

def load_and_preprocces_twitter_task_b(MAX_BRANCH_LENGTH_task_b):

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_taskb_extraction_loop)

    # d_tw = load_reddit_data()
    # tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_taskb_extraction_loop)

    ##########################################################################################################
    # this is simply for original branch y_train and y_test values
    tr_x_b, tr_y_b, _, _, dv_x_b, dv_y_b = branchify_data(d_tw, branchify_twitter_extraction_loop)
    MAX_BRANCH_LENGTH = MAX_BRANCH_LENGTH_task_b #max(len(max(dv_x_b, key=len)), len(max(tr_x_b, key=len)))
    #print('computed MAX_BRANCH_LENGTH =', MAX_BRANCH_LENGTH)
    y_train_b = transform_labels(tr_y_b, MAX_BRANCH_LENGTH)
    y_test_b = transform_labels(dv_y_b, MAX_BRANCH_LENGTH)
    ##########################################################################################################

    #MAX_BRANCH_LENGTH_task_b = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    #print('computed MAX_BRANCH_LENGTH_task_b=', MAX_BRANCH_LENGTH_task_b)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)
    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)

    #### feature engineering
    
    for new_feature_f in [kfold_feature]:
        x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train_b,
                                             None, y_test_b)
        print('x_train.shape', x_train.shape)
    
    ####
    y_train = tr_y
    y_test = dv_y

    return x_train, x_test, y_train, y_test, len(y_test)


def load_and_preprocces_reddit_task_b(MAX_BRANCH_LENGTH_task_b):

    d_tw = load_reddit_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_taskb_extraction_loop)

    # d_tw = load_reddit_data()
    # tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_taskb_extraction_loop)

    ##########################################################################################################
    # this is simply for original branch y_train and y_test values
    tr_x_b, tr_y_b, _, _, dv_x_b, dv_y_b = branchify_data(d_tw, branchify_reddit_extraction_loop)
    MAX_BRANCH_LENGTH = MAX_BRANCH_LENGTH_task_b #max(len(max(dv_x_b, key=len)), len(max(tr_x_b, key=len)))
    #print('computed MAX_BRANCH_LENGTH =', MAX_BRANCH_LENGTH)
    y_train_b = transform_labels(tr_y_b, MAX_BRANCH_LENGTH)
    y_test_b = transform_labels(dv_y_b, MAX_BRANCH_LENGTH)
    ##########################################################################################################

    #MAX_BRANCH_LENGTH_task_b = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    #print('computed MAX_BRANCH_LENGTH_task_b=', MAX_BRANCH_LENGTH_task_b)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)
    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)

    #### feature engineering
    
    for new_feature_f in [kfold_feature_reddit]:
        x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train_b,
                                             None, y_test_b)
        print('x_train.shape', x_train.shape)

    return x_train, x_test, tr_y, dv_y


def calculate_max_length():

    d_tw = load_reddit_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_taskb_extraction_loop)

    MAX_BRANCH_LENGTH_REDDIT = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH=', MAX_BRANCH_LENGTH_REDDIT)

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_taskb_extraction_loop)

    MAX_BRANCH_LENGTH_TWITTER = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH=', MAX_BRANCH_LENGTH_TWITTER)

    return max([MAX_BRANCH_LENGTH_REDDIT, MAX_BRANCH_LENGTH_TWITTER])


def combine_data():
    MAX_BRANCH_LENGTH = calculate_max_length()
    x_train, x_test, y_train, y_test, len_twitter_test = load_and_preprocces_twitter_task_b(MAX_BRANCH_LENGTH)
    x_train_reddit, x_test_reddit, y_train_reddit, y_test_reddit = load_and_preprocces_reddit_task_b(MAX_BRANCH_LENGTH)

    x_train = np.concatenate((x_train, x_train_reddit), axis=0)
    x_test = np.concatenate((x_test, x_test_reddit), axis=0)
    y_train = np.concatenate((y_train, y_train_reddit), axis=0)
    y_test = np.concatenate((y_test, y_test_reddit), axis=0)

    return x_train, x_test, y_train, y_test, len_twitter_test

def main():
    x_train, x_test, y_train, y_test, len_twitter_test = combine_data()

    model = Sequential()
    model.add(Bidirectional(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=False), input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Activation('sigmoid'))
    model.add(Dense(NUMBER_OF_CLASSES_task_b))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    y_train, y_test = transform_labels_source(y_train, y_test, number_of_classes=NUMBER_OF_CLASSES_task_b)

    model.fit(x_train, y_train, nb_epoch=8, batch_size=64)  # nb_epoch=50

    preds_test = model.predict(x_test, 100)

    x_test_twitter = x_test[:len_twitter_test]
    preds_test_twitter = preds_test[:len_twitter_test]
    y_test_twitter = y_test[:len_twitter_test]

    preds_test_twitter = [np.argmax(xx) for xx in preds_test_twitter]   #includes predictions for padded data
    y_test_twitter = [np.argmax(yy) for yy in y_test_twitter]

    xx_prev0 = None
    br = 0
    preds_test2 = []
    y_test2 = []
    #  predictions for the same source will be printed and separated by newline
    for i, xx in enumerate(x_test_twitter):
        if (xx[0][:200] != xx_prev0).any():
            br += 1
            preds_test2.append(preds_test_twitter[i])
            y_test2.append(y_test_twitter[i])
        
        xx_prev0 = xx[0][:200]

    print('br: ', br)
    print('len: ', len(preds_test2))
    print('real accuracy_score(preds_test, y_test)', accuracy_score(preds_test2, y_test2))


main()
