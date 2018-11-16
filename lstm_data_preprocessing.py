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

_DATA_DIR = "data"      # directory where twitter.pkl and reddit.pkl
                                                                        # are located

# comments:
    # see main for usage examples
    # - what to do with ids that are not found (currently nothing, ignored, see first twitter training in
  # print_structure_example())
    # - source posts don't all have the same label (e.g. query, support)
    # - from Turing paper, this is not yet implemented: Since there is overlap between
            # branches originating from the same source
            # tweet, we exclude the repeating tweets from the
            # loss function using a mask at the training stage.

def load_twitter_data():
    """Loads twitter dataset in twitter.pkl"""

    os.chdir(_DATA_DIR)
    data = pd.read_pickle('twitter.pkl')

    return data


def load_reddit_data():
    """Loads reddit dataset in reddit.pkl"""

    os.chdir(_DATA_DIR)
    data = pd.read_pickle('reddit.pkl')

    return data


def branchify_data(data, branchify_function):
    """Create branches from loaded dataset"""

    # train
    train_branches_texts = []
    train_branches_labels = []
    branchify_function(data['train'], train_branches_texts, train_branches_labels)

    # test
    test_branches_texts = []
    test_branches_labels = []
    branchify_function(data['test'], test_branches_texts, test_branches_labels)

    # dev
    dev_branches_texts = []
    dev_branches_labels = []
    branchify_function(data['dev'], dev_branches_texts, dev_branches_labels)

    return train_branches_texts, train_branches_labels, test_branches_texts, test_branches_labels, \
            dev_branches_texts, dev_branches_labels


def branchify_twitter_extraction_loop(data, branches_texts, branches_labels):
    """Extract tweets from ids of branches"""

    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            branch_texts = []
            branch_labels = []

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != str(source_text['source']['id']):
                        print(source_text['source']['id_str'], source_text['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branch_texts.append(source_text['source']['text'])
                    branch_labels.append(source_text['source']['label'])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)

                        branch_texts.append(reply['text'])
                        branch_labels.append(reply['label'])

            branches_texts.append(branch_texts)
            branches_labels.append(branch_labels)


def branchify_reddit_extraction_loop(data, branches_texts, branches_labels):
    """Extract reddit posts from ids of branches"""

    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            branch_texts = []
            branch_labels = []

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branch_texts.append(source_text['source']['text'])
                    branch_labels.append(source_text['source']['label'])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:

                        branch_texts.append(reply['text'])
                        branch_labels.append(reply['label'])

            branches_texts.append(branch_texts)
            branches_labels.append(branch_labels)


def print_structure_example():
    """Print dataset structure examples for both datasets"""

    d_tw = load_twitter_data()
    d_rd = load_reddit_data()

    print('d_tw.keys()', d_tw.keys())
    print('d_rd.keys()', d_rd.keys())

    print('(d_tw[train][0].keys()', d_tw['train'][0].keys())
    print('(d_rd[train][0].keys()', d_rd['train'][0].keys())

    print('d_tw[train][0][source].keys()', d_tw['train'][0]['source'].keys())
    print('d_rd[train][0][source].keys()', d_rd['train'][0]['source'].keys())

    print('d_tw[train][0][structure]', d_tw['train'][0]['structure'])
    print('d_rd[train][0][structure]', d_rd['train'][0]['structure'])

    print('d_tw[train][0][branches]', d_tw['train'][0]['branches'])
    print('d_rd[train][0][branches]', d_rd['train'][0]['branches'])

    print('d_tw[train][0][replies][0].keys()', d_tw['train'][0]['replies'][0].keys())
    print('d_rd[train][0][replies][0].keys()', d_rd['train'][0]['replies'][0].keys())

    print('for reply in d_tw[train][0][replies]:')
    for reply in d_tw['train'][0]['replies']:
        print('reply[id_str]', reply['id_str'])
        print('reply[id]', reply['id'])

    print('for reply in d_rd[train][0][replies]:')
    for reply in d_rd['train'][0]['replies']:
        print('reply[id_str]', reply['id_str'])


def class_to_onehot(y, max_y = None):
    if max_y == None:
        max_y = max(y)
    y_oh = np.zeros((len(y), max_y + 1))
    y_oh[range(len(y)), y] = 1
    return y_oh


def main():

    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    #d_re = load_reddit_data()
    #tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_re, branchify_reddit_extraction_loop)

    MAX_BRANCH_LENGTH = 25
    NUMBER_OF_CLASSES = 4
    GLOVE_DIR = 'C:\\Users\\viktor\\Projects\\Python\\projektHSP\\glove.twitter.27B\\glove.twitter.27B.200d.txt'

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

    x_train = []
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
                        if embeddings_index[word][1] > 113 or embeddings_index[word][1] < -113:
                            print('treca')
                            print(embeddings_index[word])
                        #print(word)
                        for broj in embeddings_index[word]:
                            suma.append(broj)
                        suma = np.asarray(suma)
                        #print(embeddings_index[word])
                    else:
                        suma += embeddings_index[word]
                    br += 1
            #print(suma)
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
        x_train.append(temp_podatak)
    x_train = np.asarray(x_train)
    #print(x_train[0])

    from sklearn.externals import joblib
    le = joblib.load('le.pkl')
    y_train = []
    for yy in tr_y:
        yy = le.transform(yy)
        temp = class_to_onehot(yy, NUMBER_OF_CLASSES)

        if temp.shape[0] < MAX_BRANCH_LENGTH:
            to_add = np.zeros((MAX_BRANCH_LENGTH - temp.shape[0], temp.shape[1]))
            to_add[:, NUMBER_OF_CLASSES] = 1
            temp = np.concatenate((temp, to_add), axis = 0)
        else:
            temp = temp[:MAX_BRANCH_LENGTH]

        y_train.append(temp)
    y_train = np.asarray(y_train)

    print('test')
    x_test = []
    for podatak in dv_x:
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
                        if embeddings_index[word][1] > 113 or embeddings_index[word][1] < -113:
                            print('treca')
                            print(embeddings_index[word])
                        #print(word)
                        for broj in embeddings_index[word]:
                            suma.append(broj)
                        suma = np.asarray(suma)
                        #print(embeddings_index[word])
                    else:
                        suma += embeddings_index[word]
                    br += 1
            #print(suma)
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
        x_test.append(temp_podatak)
    x_test = np.asarray(x_test)

    from sklearn.externals import joblib
    le = joblib.load('le.pkl')
    y_test = []
    for yy in dv_y:
        yy = le.transform(yy)
        temp = class_to_onehot(yy, NUMBER_OF_CLASSES)

        if temp.shape[0] < MAX_BRANCH_LENGTH:
            to_add = np.zeros((MAX_BRANCH_LENGTH - temp.shape[0], temp.shape[1]))
            to_add[:, NUMBER_OF_CLASSES] = 1
            temp = np.concatenate((temp, to_add), axis = 0)
        else:
            temp = temp[:MAX_BRANCH_LENGTH]

        y_test.append(temp)
    y_test = np.asarray(y_test)

    np.save('x_train', x_train)
    np.save('x_test', x_test)
    np.save('y_train', y_train)
    np.save('y_test', y_test)
    

main()

    
    