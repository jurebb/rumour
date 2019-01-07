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

    MAX_BRANCH_LENGTH_task_b = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH_task_b=', MAX_BRANCH_LENGTH_task_b)

    embeddings_index = make_embeddings_index()

    x_train_temp = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)
    # y_train_temp = transform_labels(tr_y, MAX_BRANCH_LENGTH_task_b)

    x_test_temp = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH_task_b)
    # y_test_temp = transform_labels(dv_y, MAX_BRANCH_LENGTH_task_b)

    twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    #### feature engineering
    # for new_feature_f in [reddit_kind_feature, reddit_used_feature, reddit_id_str_feature, reddit_score_feature, reddit_controversiality_feature]:
    # for new_feature_f in [twitter_user_mention_count_feature]:
    #     x_train, _, x_test = concat_features([new_feature_f], x_train_temp, None, x_test_temp, y_train_temp, None, y_test_temp)
        # y_train = y_train_temp
        # y_test = y_test_temp
        # print('x_train.shape', x_train.shape)

    ####

    x_train = x_train_temp
    x_test = x_test_temp
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
    preds_test = [np.argmax(xx) for x in preds_test for xx in x] #includes predictions for padded data
    print('len(y_test)', len(y_test))
    print('accuracy_score(preds_test, y_test)', accuracy_score(preds_test, y_test))


main()
