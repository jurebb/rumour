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

def load_and_preprocces_twitter2(MAX_BRANCH_LENGTH, add_features):

    d_tw = load_twitter_data()
    tr_x, tr_y, test_x, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_dev = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_dev = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    x_test = transform_data(test_x, embeddings_index, MAX_BRANCH_LENGTH) 

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)
    if add_features:
        #### feature engineering
        for new_feature_f in [twitter_favorite_count_feature]:
            x_train, x_test, x_dev = concat_features([new_feature_f], x_train, x_test, x_dev, y_train, [], y_dev)
            print('x_train.shape', x_train.shape)

        #x_train = np.concatenate((x_train, np.ones((x_train.shape[0], x_train.shape[1], 1))), axis=2)  #add is_twitter feature
        #x_test = np.concatenate((x_test, np.ones((x_test.shape[0], x_test.shape[1], 1))), axis=2)  #add is_twitter feature

    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)

    return x_train, x_test, y_train, [], len(x_test)

def load_and_preprocces_reddit2(MAX_BRANCH_LENGTH, add_features):
    d_tw = load_reddit_data()
    tr_x, tr_y, test_x, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    print('GLAVNOOOOOOOOOOOO2: ', len(test_x))

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH) 

    x_dev = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH) 
    y_dev = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #twitter_user_description_feature = twitter_user_description_feature_(embeddings_index)
    x_test = transform_data(test_x, embeddings_index, MAX_BRANCH_LENGTH) 

    if add_features:
        #### feature engineering
        for new_feature_f in [reddit_score_feature]:
            x_train, x_test, x_dev = concat_features([new_feature_f], x_train, x_test, x_dev, y_train, [], y_dev)
            print('x_train.shape', x_train.shape)

        #x_train = np.concatenate((x_train, np.zeros((x_train.shape[0], x_train.shape[1], 1))), axis=2) #instead of twitter_user_mention_count_feature
        #x_test = np.concatenate((x_test, np.zeros((x_test.shape[0], x_test.shape[1], 1))), axis=2) #instead of twitter_user_mention_count_feature

    x_train = np.concatenate((x_train, x_dev), axis=0)
    y_train = np.concatenate((y_train, y_dev), axis=0)

    return x_train, x_test, y_train, []

def combine_data(MAX_BRANCH_LENGTH, add_features=True):

    x_train, x_test, y_train, _, len_twitter_test = load_and_preprocces_twitter2(MAX_BRANCH_LENGTH, add_features)
    x_train_reddit, x_test_reddit, y_train_reddit, _ = load_and_preprocces_reddit2(MAX_BRANCH_LENGTH, add_features)

    x_train = np.concatenate((x_train, x_train_reddit), axis=0)
    x_test = np.concatenate((x_test, x_test_reddit), axis=0)
    y_train = np.concatenate((y_train, y_train_reddit), axis=0)

    return x_train, x_test, y_train, len_twitter_test

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

def remove_padded_data2(pred, x_test):

    x_test_flat = []
    for i in range(len(x_test)):
        for j in range(len(x_test[i])):
            x_test_flat.append(x_test[i][j])
    pred2 = []
    for i in range(len(pred)):
        #print('x_test200', x_test_flat[i][:200])
        if (x_test_flat[i][:200] != np.zeros(200)).any():
            pred2.append(pred[i])

    return pred2


def get_predictions_combined_task_a():
    MAX_BRANCH_LENGTH = calculate_max_length()
    x_train, x_test, y_train, len_twitter_test = combine_data(MAX_BRANCH_LENGTH)

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
    
    preds_test_twitter = [np.argmax(xx) for x in preds_test_twitter for xx in x] #includes predictions for padded data

    preds_test_twitter = remove_padded_data2(preds_test_twitter, x_test[:len_twitter_test])
    
    d_tw = load_twitter_data()
    tr_x, tr_y, te_x, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    preds_test_twitter = remove_duplicated_data2(preds_test_twitter, d_tw, te_x)

    print("REDDIT")
    
    preds_test_reddit = preds_test[len_twitter_test:]

    preds_test_reddit = [np.argmax(xx) for x in preds_test_reddit for xx in x] #includes predictions for padded data
    print('predikcija1 ', preds_test_reddit)
    preds_test_reddit = remove_padded_data2(preds_test_reddit, x_test[len_twitter_test:])
    print('predikcija2 ', preds_test_reddit)
    d_tw = load_reddit_data()
    tr_x, tr_y, te_x, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    preds_test_reddit = remove_duplicated_data2(preds_test_reddit, d_tw, te_x)

    return preds_test_twitter, preds_test_reddit


def submit_task_a(tw_pkl, rd_pkl):
    preds_test_twitter, preds_test_reddit = get_predictions_combined_task_a()

    data = load_twitter_data()

    x_id_tw = []
    counter = dict()
    br = 0
    for base_text in data['test']:
        x_id_tw.append(base_text['source']['id'])

        for reply in base_text['replies']:
            if pd.notnull(reply['text']) and reply['text'] != '[deleted]' and reply['text'] != '[removed]':
                x_id_tw.append(reply['id'])
        
    data = load_reddit_data()

    x_id_rd = []

    for base_text in data['test']:
        x_id_rd.append(base_text['source']['id_str'])

        for reply in base_text['replies']:
            # if pd.notnull(reply['text']) and reply['text'] != '[deleted]' and reply['text'] != '[removed]':
            x_id_rd.append(reply['id_str'])

    submit_dict_taskaenglish = dict()

    le = joblib.load('le.pkl')
    print('predikcija3 ', preds_test_reddit)
    preds_test_twitter = le.inverse_transform(preds_test_twitter)
    preds_test_reddit = le.inverse_transform(preds_test_reddit)

    for i in range(len(x_id_tw)):
        submit_dict_taskaenglish[x_id_tw[i]] = preds_test_twitter[i]
    for i in range(len(x_id_rd)):
        submit_dict_taskaenglish[x_id_rd[i]] = preds_test_reddit[i]

    print('len(x_id_rd)', len(x_id_rd))
    print('len(preds_test_reddit)', len(preds_test_reddit))

    print('len(x_id_tw)', len(x_id_tw))
    print('len(preds_test_twitter)', len(preds_test_twitter))

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