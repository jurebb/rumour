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

import keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

from sequential.prepare_seq_data import *
from sequential.additional_features import *
from sequential.additional_features_reddit import *
from sequential.additional_computed_features import *
from sequential.additional_computed_features_reddit import *
from sequential.feature_utils import *

import task_b.branch_model_combined as task_b

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


def calculate_sample_weights_task_a(y_train, MAX_BRANCH_LENGTH):
    """for sample_weights in keras, because metric now is f1 and dataset is highly unbalanced"""

    classes_count = dict()

    counter_all = 0

    for branch in y_train:
        for timestep in branch:
            if str(timestep) not in classes_count.keys():
                classes_count[str(timestep)] = 0
            else:
                classes_count[str(timestep)] += 1

            counter_all += 1

    class_weights = dict()
    sorted_keys = sorted(classes_count.keys())
    for key in sorted_keys:
        # class_weights[key] = (1 - (classes_count[key] / counter_all)) # * 10
        class_weights[key] = min([9.0, (counter_all / (5 * classes_count[key]))])

    ##### label DENY to high weight
    # class_weights['[0. 0. 0. 1.]'] = 5
    print('class_weights', class_weights)
    #####

    y_weights = []
    for branch in y_train:
        branch_weigths = []
        for timestep in branch:
            branch_weigths.append(class_weights[str(timestep)])

        y_weights.append(np.asarray(branch_weigths))

    return np.asarray(y_weights)


def load_and_preprocces_twitter(MAX_BRANCH_LENGTH, add_features):

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_test = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)
    if add_features:
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

def load_and_preprocces_twitter2(MAX_BRANCH_LENGTH, add_features):

    d_tw = load_twitter_data()
    tr_x, tr_y, test_x, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    print('GLAVNOOOOOOOOOOOO: ', len(test_x))

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_test = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)
    if add_features:
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

def load_and_preprocces_reddit(MAX_BRANCH_LENGTH, add_features):
    d_tw = load_reddit_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_test = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)

    if add_features:
        #### feature engineering
        for new_feature_f in [reddit_score_feature, reddit_likes_feature, reddit_punctuation_count_feature('?'), 
            reddit_word_counter_feature, reddit_url_counter_feature, 
            reddit_previous_post_similarity_feature(x_train, x_test, y_train, y_test, embeddings_index)]:
            x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train, None, y_test)
            print('x_train.shape', x_train.shape)

        x_train = np.concatenate((x_train, np.zeros((x_train.shape[0], x_train.shape[1], 1))), axis=2) #instead of twitter_user_mention_count_feature
        x_test = np.concatenate((x_test, np.zeros((x_test.shape[0], x_test.shape[1], 1))), axis=2) #instead of twitter_user_mention_count_feature

    return x_train, x_test, y_train, y_test

def combine_data(MAX_BRANCH_LENGTH, add_features=True):

    x_train, x_test, y_train, y_test, len_twitter_test = load_and_preprocces_twitter2(MAX_BRANCH_LENGTH, add_features)
    x_train_reddit, x_test_reddit, y_train_reddit, y_test_reddit = load_and_preprocces_reddit(MAX_BRANCH_LENGTH, add_features)

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
    x_train, x_test, y_train, y_test, len_twitter_test = combine_data(MAX_BRANCH_LENGTH, add_features=False)

    best_f1 = 0
    best_params = [0, 0, 0, 0, 0]
    for units in [32, 64, 100, 128, 256]:
        for dropout in [0.01, 0.05, 0.1, 0.2, 0.4]:
            for recurrent_dropout in [0.01, 0.05, 0.1, 0.2, 0.4]:
                for lr in [0.0001, 0.001, 0.01]:
                    for nb_epoch in [4, 8, 16, 32]:

                        sample_weights = calculate_sample_weights_task_a(y_train, MAX_BRANCH_LENGTH)

                        model = Sequential()
                        model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                                        input_shape=(x_train.shape[1], x_train.shape[2])))
                        model.add(Activation('sigmoid'))
                        model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
                        model.add(Activation('softmax'))

                        adam = Adam(lr=0.001)
                        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'], sample_weight_mode='temporal')
                        model.fit(x_train, y_train, nb_epoch=8, batch_size=64, sample_weight=sample_weights, verbose=0) 
                        preds_test = model.predict(x_test, 100)
                        preds_test_twitter = preds_test[:len_twitter_test]
                        y_test_twitter = y_test[:len_twitter_test]

                        preds_test_twitter = [np.argmax(xx) for x in preds_test_twitter for xx in x] 
                        y_test_twitter = [np.argmax(xx) if np.max(xx) != 0 else 'None' for x in y_test_twitter for xx in x] 
                        
                        preds_test_twitter, y_test_twitter = remove_padded_data(preds_test_twitter, y_test_twitter)

                        d_tw = load_twitter_data()
                        _, _, _, _, dv_x, _ = branchify_data(d_tw, branchify_twitter_extraction_loop)

                        preds_test_twitter, y_test_twitter = remove_duplicated_data(preds_test_twitter, y_test_twitter, d_tw, dv_x)

                        curr_score = f1_score(preds_test_twitter, y_test_twitter, average='macro')
                        print('f1_score(preds_test_twitter, y_test_twitter) after removing padded/duplicated', curr_score)

                        keras.backend.clear_session()
                                    
                        if curr_score > best_f1:
                            best_f1 = curr_score
                            best_params = [units, dropout, recurrent_dropout, lr, nb_epoch]
    print('FINAL F1')
    print(best_f1)
    print('units, dropout, recurrent_dropout, lr, nb_epoch')
    print(best_params)


def get_predictions_combined_task_a():
    MAX_BRANCH_LENGTH = calculate_max_length()
    x_train, x_test, y_train, y_test, len_twitter_test = combine_data(MAX_BRANCH_LENGTH)

    sample_weights = calculate_sample_weights_task_a(y_train, MAX_BRANCH_LENGTH)

    model = Sequential()
    model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Activation('sigmoid'))
    model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
    model.add(Activation('softmax'))

    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'], sample_weight_mode='temporal',
                  )
    model.fit(x_train, y_train, nb_epoch=8, batch_size=64, sample_weight=sample_weights)  # nb_epoch=50

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

    return preds_test_twitter, preds_test_reddit, y_test_twitter, y_test_reddit


def submit_task_a(tw_pkl, rd_pkl):
    preds_test_twitter, preds_test_reddit, y_test_twitter, y_test_reddit = get_predictions_combined_task_a()

    data = pd.read_pickle(tw_pkl)

    x_id_tw = []
    counter = dict()
    br = 0
    for base_text in data['dev']:
        x_id_tw.append(base_text['source']['id'])
        label = base_text['source']['label']
        if label in counter:
            counter[label] += 1
        else:
            counter[label] = 1
        br += 1

        for reply in base_text['replies']:
            if pd.notnull(reply['text']) and reply['text'] != '[deleted]' and reply['text'] != '[removed]':
                x_id_tw.append(reply['id'])
                label = reply['label']
                if label in counter:
                    counter[label] += 1
                else:
                    counter[label] = 1
                br += 1
    print(counter)
    print(br)
    for key in counter:
        counter[key] /= br
    print(counter)
        
    data = pd.read_pickle(rd_pkl)

    x_id_rd = []

    print('data.keys()', data.keys())
    print('data[test][0].keys()', data['dev'][0].keys())
    for base_text in data['dev']:
        x_id_rd.append(base_text['source']['id_str'])

        for reply in base_text['replies']:
            # if pd.notnull(reply['text']) and reply['text'] != '[deleted]' and reply['text'] != '[removed]':
            x_id_rd.append(reply['id_str'])

    submit_dict_taskaenglish = dict()

    le = joblib.load('le.pkl')
    preds_test_twitter = le.inverse_transform(preds_test_twitter)
    preds_test_reddit = le.inverse_transform(preds_test_reddit)

    for i in range(len(x_id_tw)):
        submit_dict_taskaenglish[x_id_tw[i]] = preds_test_twitter[i]
    for i in range(len(x_id_rd)):
        submit_dict_taskaenglish[x_id_rd[i]] = preds_test_reddit[i]

    print('len(x_id_rd)', len(x_id_rd))
    print('len(preds_test_reddit)', len(preds_test_reddit))

    assert len(x_id_tw) == len(preds_test_twitter)
    assert len(x_id_rd) == len(preds_test_reddit)

    # submit_dict_taskaenglish = json.dumps(submit_dict_taskaenglish)
    # f = open('answer', 'w')
    # f.write('blin')
    # f.write(submit_dict_taskaenglish)
    #
    # print('submit_dict_taskaenglish', submit_dict_taskaenglish)

    return submit_dict_taskaenglish


def myconverter(o):
    if isinstance(o, np.float32):
        return float(o)


def submit_json():
    submit_dict_taskaenglish = submit_task_a(tw_pkl='twitter_new2.pkl', rd_pkl='reddit_new2.pkl')
    submit_dict_taskbenglish = task_b.submit_task_b(tw_pkl='twitter_new2.pkl', rd_pkl='reddit_new2.pkl')

    final_json = dict()
    final_json['subtaskaenglish'] = submit_dict_taskaenglish
    final_json['subtaskbenglish'] = submit_dict_taskbenglish
    final_json['subtaskadanish'] = None
    final_json['subtaskbdanish'] = None
    final_json['subtaskarussian'] = None
    final_json['subtaskbrussian'] = None

    f = open('answer_full2.json', 'w')
    final_json = json.dumps(final_json, default=myconverter)
    f.write(final_json)


if __name__ == "__main__":
    submit_json()