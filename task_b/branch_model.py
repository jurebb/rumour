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


def main():
    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_taskb_extraction_loop)

    # d_tw = load_reddit_data()
    # tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_taskb_extraction_loop)

    ##########################################################################################################
    # this is simply for original branch y_train and y_test values
    tr_x_b, tr_y_b, _, _, dv_x_b, dv_y_b = branchify_data(d_tw, branchify_twitter_extraction_loop)
    MAX_BRANCH_LENGTH = max(len(max(dv_x_b, key=len)), len(max(tr_x_b, key=len)))
    print('computed MAX_BRANCH_LENGTH =', MAX_BRANCH_LENGTH)
    y_train_b = transform_labels(tr_y_b, MAX_BRANCH_LENGTH)
    y_test_b = transform_labels(dv_y_b, MAX_BRANCH_LENGTH)
    ##########################################################################################################

    MAX_BRANCH_LENGTH_task_b = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH_task_b=', MAX_BRANCH_LENGTH_task_b)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)
    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)

    #### feature engineering
    for new_feature_f in [twitter_favorite_count_feature, twitter_retweet_count_feature,
                          twitter_punctuation_count_feature('?'),
                          twitter_word_counter_feature, twitter_url_counter_feature,
                          twitter_previous_tweet_similarity_feature(x_train, x_test, y_train_b, y_test_b, embeddings_index),
                          twitter_user_mention_count_feature,
                          kfold_feature
                          ]:
        x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train_b,
                                             None, y_test_b)
        print('x_train.shape', x_train.shape)

    ####

    y_train = tr_y
    y_test = dv_y

    print('x_train.shape', x_train.shape)

    model = Sequential()
    model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Activation('sigmoid'))
    model.add(Dense(NUMBER_OF_CLASSES_task_b))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    y_train, y_test = transform_labels_source(y_train, y_test, number_of_classes=NUMBER_OF_CLASSES_task_b)

    model.fit(x_train, y_train, nb_epoch=8, batch_size=64)  # nb_epoch=50

    preds_test = model.predict(x_test, 100)
    print('preds_test', preds_test)
    preds_test = [np.argmax(xx) for xx in preds_test]   #includes predictions for padded data
    print('preds_test', preds_test)
    print('len(preds_test)', len(preds_test))
    print('preds_test[0]', preds_test[0])
    y_test = [np.argmax(yy) for yy in y_test]
    print('y_test', y_test)
    print('len(y_test)', len(y_test))
    print('y_test[0]', y_test[0])
    print('accuracy_score(preds_test, y_test)', accuracy_score(preds_test, y_test))


main()
