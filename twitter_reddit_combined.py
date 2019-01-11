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
from sequential.additional_features_reddit import *
from sequential.additional_computed_features import *
from sequential.additional_computed_features_reddit import *
from sequential.feature_utils import *

from numpy.random import seed
from tensorflow import set_random_seed
seed(12)
set_random_seed(22)

_DATA_DIR = "C:\\Users\\viktor\\Projects\\Python\\data_set\\data"
MAX_BRANCH_LENGTH = -1
NUMBER_OF_CLASSES = 4
#GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.840B.300d\\glove.840B.300d.txt'
GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'
#  GLOVE_DIR = '/home/interferon/Documents/dipl_projekt/glove/glove.twitter.27B.200d.txt'

def load_and_preprocces_twitter(MAX_BRANCH_LENGTH):

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_test = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    #### feature engineering
    for new_feature_f in [twitter_favorite_count_feature, twitter_retweet_count_feature, twitter_punctuation_count_feature('?'), 
            twitter_word_counter_feature, twitter_url_counter_feature, 
            twitter_previous_tweet_similarity_feature(x_train, x_test, y_train, y_test, embeddings_index),
            twitter_user_mention_count_feature]:
        x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train, None, y_test)
        print('x_train.shape', x_train.shape)

    #x_train = np.concatenate((x_train, np.ones((x_train.shape[0], x_train.shape[1], 1))), axis=2)  #add is_twitter feature
    #x_test = np.concatenate((x_test, np.ones((x_test.shape[0], x_test.shape[1], 1))), axis=2)  #add is_twitter feature

    return x_train, x_test, y_train, y_test, len(y_test)

def load_and_preprocces_reddit(MAX_BRANCH_LENGTH):
    d_tw = load_reddit_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_test = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    #### feature engineering
    for new_feature_f in [reddit_score_feature, reddit_likes_feature, reddit_punctuation_count_feature('?'), 
        reddit_word_counter_feature, reddit_url_counter_feature, 
        reddit_previous_post_similarity_feature(x_train, x_test, y_train, y_test, embeddings_index)]:
        x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train, None, y_test)
        print('x_train.shape', x_train.shape)

    x_train = np.concatenate((x_train, np.zeros((x_train.shape[0], x_train.shape[1], 1))), axis=2) #instead of twitter_user_mention_count_feature
    x_test = np.concatenate((x_test, np.zeros((x_test.shape[0], x_test.shape[1], 1))), axis=2) #instead of twitter_user_mention_count_feature

    return x_train, x_test, y_train, y_test

def combine_data(MAX_BRANCH_LENGTH):

    x_train, x_test, y_train, y_test, len_twitter_test = load_and_preprocces_twitter(MAX_BRANCH_LENGTH)
    x_train_reddit, x_test_reddit, y_train_reddit, y_test_reddit = load_and_preprocces_reddit(MAX_BRANCH_LENGTH)

    x_train = np.concatenate((x_train, x_train_reddit), axis=0)
    x_test = np.concatenate((x_test, x_test_reddit), axis=0)
    y_train = np.concatenate((y_train, y_train_reddit), axis=0)
    y_test = np.concatenate((y_test, y_test_reddit), axis=0)

    return x_train, x_test, y_train, y_test, len_twitter_test

def calculate_max_length():

    d_tw = load_reddit_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    MAX_BRANCH_LENGTH_REDDIT = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH=', MAX_BRANCH_LENGTH_REDDIT)

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    MAX_BRANCH_LENGTH_TWITTER = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH=', MAX_BRANCH_LENGTH_TWITTER)

    return max([MAX_BRANCH_LENGTH_REDDIT, MAX_BRANCH_LENGTH_TWITTER])


def lstm_hyperparameters():
    MAX_BRANCH_LENGTH = calculate_max_length()
    x_train, x_test, y_train, y_test, len_twitter_test = combine_data(MAX_BRANCH_LENGTH)

    best_acc = 0
    best_params = [0, 0, 0, 0, 0]
    for units in [32, 64, 100, 128, 256]:
        for dropout in [0.01, 0.05, 0.1, 0.2, 0.4]:
            for recurrent_dropout in [0.01, 0.05, 0.1, 0.2, 0.4]:
                for lr in [0.0001, 0.001, 0.01]:
                    for nb_epoch in [4, 8, 16, 32]:

                        model = Sequential()
                        model.add(LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout, 
                            return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
                        model.add(Activation('sigmoid'))
                        model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
                        model.add(Activation('softmax'))

                        adam = Adam(lr=lr)
                        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
                        model.fit(x_train, y_train, nb_epoch=nb_epoch, batch_size=64, verbose=0)  # nb_epoch=50

                        preds_test = model.predict(x_test, 100)
                        preds_test_twitter = preds_test[:len_twitter_test]
                        y_test_twitter = y_test[:len_twitter_test]

                        preds_test_twitter = [np.argmax(xx) for x in preds_test_twitter for xx in x] 
                        y_test_twitter = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test_twitter for xx in x] 
                        
                        preds_test_twitter, y_test_twitter = remove_padded_data(preds_test_twitter, y_test_twitter)

                        d_tw = load_twitter_data()
                        _, _, _, _, dv_x, _ = branchify_data(d_tw, branchify_twitter_extraction_loop)

                        preds_test_twitter, y_test_twitter = remove_duplicated_data(preds_test_twitter, y_test_twitter, d_tw, dv_x)

                        acc_score = accuracy_score(preds_test_twitter, y_test_twitter)
                        print('accuracy_score(preds_test_twitter, y_test_twitter) after removing padded/duplicated', acc_score)
                                    
                        if acc_score > best_acc:
                            best_acc = acc_score
                            best_params = [units, dropout, recurrent_dropout, lr, nb_epoch]
    print('FINAL ACC')
    print(best_acc)
    print('units, dropout, recurrent_dropout, lr, nb_epoch')
    print(best_params)

def main():
    MAX_BRANCH_LENGTH = calculate_max_length()
    x_train, x_test, y_train, y_test, len_twitter_test = combine_data(MAX_BRANCH_LENGTH)

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

    print("TWITTER")

    preds_test_twitter = preds_test[:len_twitter_test]
    y_test_twitter = y_test[:len_twitter_test]

    preds_test_twitter = [np.argmax(xx) for x in preds_test_twitter for xx in x] #includes predictions for padded data
    y_test_twitter = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test_twitter for xx in x] #predictions for padded data added as str None
    print('len(y_test_twitter)', len(y_test_twitter))
    print('accuracy_score(preds_test_twitter, y_test_twitter)', accuracy_score(preds_test_twitter, y_test_twitter))

    preds_test_twitter, y_test_twitter = remove_padded_data(preds_test_twitter, y_test_twitter)

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    preds_test_twitter, y_test_twitter = remove_duplicated_data(preds_test_twitter, y_test_twitter, d_tw, dv_x)

    print('accuracy_score(preds_test_twitter, y_test_twitter) after removing padded/duplicated', accuracy_score(preds_test_twitter, y_test_twitter))
    print('preds_test_twitter', preds_test_twitter)


    print("REDDIT")
    
    preds_test_reddit = preds_test[len_twitter_test:]
    y_test_reddit = y_test[len_twitter_test:]

    preds_test_reddit = [np.argmax(xx) for x in preds_test_reddit for xx in x] #includes predictions for padded data
    y_test_reddit = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test_reddit for xx in x] #predictions for padded data added as str None
    print('len(y_test_reddit)', len(y_test_reddit))
    print('accuracy_score(preds_test_reddit, y_test_reddit)', accuracy_score(preds_test_reddit, y_test_reddit))

    preds_test_reddit, y_test_reddit = remove_padded_data(preds_test_reddit, y_test_reddit)

    d_tw = load_reddit_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    preds_test_reddit, y_test_reddit = remove_duplicated_data(preds_test_reddit, y_test_reddit, d_tw, dv_x)

    print('accuracy_score(preds_test_reddit, y_test_reddit) after removing padded/duplicated', accuracy_score(preds_test_reddit, y_test_reddit))
    print('preds_test_reddit', preds_test_reddit)

main()
