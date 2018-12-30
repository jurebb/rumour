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

from numpy.random import seed
from tensorflow import set_random_seed
seed(12)
set_random_seed(22)

MAX_BRANCH_LENGTH = 24
NUMBER_OF_CLASSES = 4
GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
#  GLOVE_DIR = '/home/interferon/Documents/dipl_projekt/glove/glove.twitter.27B.200d.txt'



# region feature_selection
def fwd_feature_selection_twitter():
    """"prints out the best performing combination of additional concatted features for the twitter dataset.
    uses a fwd feature selection approach"""

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)
    MAX_BRANCH_LENGTH = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    embeddings_index = make_embeddings_index()
    x_train_init = transform_data(tr_x, embeddings_index)
    y_train_init = transform_labels(tr_y)
    x_test_init = transform_data(dv_x, embeddings_index)
    y_test_init = transform_labels(dv_y)

    twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    all_features = [twitter_user_description_feature, twitter_user_id_feature, twitter_retweet_count_feature,
                    twitter_profile_favourites_count, twitter_profile_use_background_image_feature,
                    twitter_time_feature, twitter_favorite_count_feature, twitter_user_mention_contains_feature,
                    twitter_user_mention_count_feature, twitter_word_counter_feature,
                    twitter_url_counter_feature, twitter_punctuation_count_feature('?'),
                    twitter_punctuation_count_feature('!'), twitter_punctuation_contains_feature('?'),
                    twitter_punctuation_contains_feature('!'),
                    twitter_previous_tweet_similarity_feature(x_train_init, x_test_init, y_train_init,
                                                            y_test_init, embeddings_index,
                                                            source_similarity=1)
                    ]

    selected_features = []
    selected_features_count = 0
    current_max_score = 0
    no_better_found = False

    x_train_current = x_train_init
    y_train_current = y_train_init
    x_test_current = x_test_init
    y_test_current = y_test_init

    while no_better_found == False:

        no_better_found = True

        for new_feature_f in all_features:

            x_train_temp, _, x_test_temp = concat_features([new_feature_f], x_train_current, None, x_test_current,
                                                           y_train_current, None, y_test_current)
            y_train = y_train_current
            y_test = y_test_current

            print('> selected_features_count', selected_features_count)
            print('> in loop x_train_temp.shape', x_train_temp.shape)

            model = Sequential()
            model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                           input_shape=(x_train_temp.shape[1], x_train_temp.shape[2])))
            model.add(Activation('sigmoid'))
            model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
            model.add(Activation('softmax'))

            adam = Adam(lr=0.001)
            model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
            model.fit(x_train_temp, y_train, nb_epoch=8, batch_size=64, verbose=0)  # nb_epoch=50

            preds_test = model.predict(x_test_temp, 100)
            preds_test = [np.argmax(xx) for x in preds_test for xx in x]  # includes predictions for padded data
            y_test = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test for xx in
                      x]  # predictions for padded data added as str None
            # print('len(y_test)', len(y_test))
            # print('accuracy_score(preds_test, y_test)', accuracy_score(preds_test, y_test))

            preds_test, y_test = remove_padded_data(preds_test, y_test)
            preds_test, y_test = remove_duplicated_data(preds_test, y_test, d_tw, dv_x)

            acc_score = accuracy_score(preds_test, y_test)
            print('accuracy_score(preds_test, y_test) after removing padded/duplicated', acc_score)
            # print('preds_test', preds_test)

            if acc_score >= current_max_score:
                print('>>> new current_max_score:', acc_score)
                print('>>> new current_max_feature:', new_feature_f)
                no_better_found = False
                current_max_score = acc_score
                current_max_score_x_train = x_train_temp
                current_max_score_x_test = x_test_temp
                current_max_feature = new_feature_f

        if no_better_found == False:
            print('>>>>>>>>>>>>>>>>>>> new better_found:', current_max_feature)
            selected_features.append(current_max_feature)
            all_features.remove(current_max_feature)
            selected_features_count += 1
            x_train_current = current_max_score_x_train
            x_test_current = current_max_score_x_test

        print('selected_features:', selected_features)
        print('max acc score:', current_max_score)


def all_subsets_feature_selection_twitter():
    """"prints out the best performing combination of additional concatted features for the twitter dataset.
    uses a fwd feature selection approach"""

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)
    MAX_BRANCH_LENGTH = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    embeddings_index = make_embeddings_index()
    x_train_init = transform_data(tr_x, embeddings_index)
    y_train_init = transform_labels(tr_y)
    x_test_init = transform_data(dv_x, embeddings_index)
    y_test_init = transform_labels(dv_y)

    twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    all_features = set([twitter_user_description_feature, twitter_user_id_feature, twitter_retweet_count_feature,
                    twitter_profile_favourites_count, twitter_profile_use_background_image_feature,
                    twitter_time_feature, twitter_favorite_count_feature,
                    twitter_user_mention_count_feature, twitter_word_counter_feature,
                    twitter_url_counter_feature, twitter_punctuation_count_feature('?'),
                    twitter_previous_tweet_similarity_feature(x_train_init, x_test_init, y_train_init,
                                                            y_test_init, embeddings_index,
                                                            source_similarity=1)
                    ])

    selected_features = []
    current_max_score = 0
    current_iteration = 0

    x_train_current = x_train_init
    y_train_current = y_train_init
    x_test_current = x_test_init
    y_test_current = y_test_init

    feature_subsets = sum(map(lambda r: list(combinations(all_features, r)), range(1, len(all_features)+1)), [])
    print('number of feature subsets: ', len(feature_subsets))
    print(feature_subsets)

    for features in feature_subsets:

        current_iteration += 1

        features_list = list(features)
        x_train_current = x_train_init
        y_train_current = y_train_init
        x_test_current = x_test_init
        y_test_current = y_test_init

        print('>>>> current feature subset:', features_list)
        print('>>>> current iteration:', current_iteration)

        for new_feature_f in features_list:

            x_train_current, _, x_test_current = concat_features([new_feature_f], x_train_current, None, x_test_current,
                                                           y_train_current, None, y_test_current)
            y_train = y_train_current
            y_test = y_test_current

        print('>>>> in loop x_train_temp.shape', x_train_current.shape)

        model = Sequential()
        model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                       input_shape=(x_train_current.shape[1], x_train_current.shape[2])))
        model.add(Activation('sigmoid'))
        model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
        model.add(Activation('softmax'))

        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(x_train_current, y_train, nb_epoch=8, batch_size=64, verbose=0)  # nb_epoch=50

        preds_test = model.predict(x_test_current, 100)
        preds_test = [np.argmax(xx) for x in preds_test for xx in x]  # includes predictions for padded data
        y_test = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test for xx in
                  x]  # predictions for padded data added as str None
        # print('len(y_test)', len(y_test))
        # print('accuracy_score(preds_test, y_test)', accuracy_score(preds_test, y_test))

        preds_test, y_test = remove_padded_data(preds_test, y_test)
        preds_test, y_test = remove_duplicated_data(preds_test, y_test, d_tw, dv_x)

        acc_score = accuracy_score(preds_test, y_test)
        print('accuracy_score(preds_test, y_test) after removing padded/duplicated', acc_score)
        # print('preds_test', preds_test)

        if acc_score >= current_max_score:
            del selected_features
            selected_features = []
            print('>>> new current_max_score:', acc_score)
            print('>>> new current_max_feature_subset:', features_list)
            current_max_score = acc_score
            selected_features = features_list
        else:
            del features_list

        print('max selected_features:', selected_features)
        print('max acc score:', current_max_score)

        del model
        del x_test_current
        del x_train_current
        del y_train_current
        del y_test_current
        del preds_test
        del y_test
        keras.backend.clear_session()
# endregion


def main():
    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    # d_tw = load_reddit_data()
    # tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    MAX_BRANCH_LENGTH = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH=', MAX_BRANCH_LENGTH)

    embeddings_index = make_embeddings_index()

    x_train_temp = transform_data(tr_x, embeddings_index) 
    y_train_temp = transform_labels(tr_y) 

    x_test_temp = transform_data(dv_x, embeddings_index) 
    y_test_temp = transform_labels(dv_y)

    twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    #### feature engineering
    # for new_feature_f in [reddit_kind_feature, reddit_used_feature, reddit_id_str_feature, reddit_score_feature, reddit_controversiality_feature]:
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
