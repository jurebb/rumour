from sequential.prepare_seq_data import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sequential.additional_features import *
import re
from html.parser import HTMLParser
from collections import Counter
from scipy import spatial


EMBEDDING_DIM = 200
NUMBER_OF_CLASSES = 4

def twitter_word_counter_feature():
    """
    length of tweets in words
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_texts = branchify_twitter_extract_feature_loop(d_tw['train'], 'text')
    dv_x_texts = branchify_twitter_extract_feature_loop(d_tw['dev'], 'text')

    for i in range(len(tr_x_texts)):
        tr_x_texts[i] = len(re.findall(r'\w+', tr_x_texts[i]))

    for i in range(len(dv_x_texts)):
        dv_x_texts[i] = len(re.findall(r'\w+', dv_x_texts[i]))

    tr_x_texts, dv_x_texts = scale(tr_x_texts, dv_x_texts)

    return tr_x_texts, None, dv_x_texts


def twitter_url_counter_feature():
    """
    number of URLs in a tweet
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_texts = branchify_twitter_extract_feature_loop(d_tw['train'], 'text')
    dv_x_texts = branchify_twitter_extract_feature_loop(d_tw['dev'], 'text')

    for i in range(len(tr_x_texts)):
        tr_x_texts[i] = len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', tr_x_texts[i]))

    for i in range(len(dv_x_texts)):
        dv_x_texts[i] = len(re.findall('https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', dv_x_texts[i]))

    tr_x_texts, dv_x_texts = scale(tr_x_texts, dv_x_texts)

    return tr_x_texts, None, dv_x_texts


def twitter_punctuation_count_feature(key='!'):
    def twitter_punctuation_count_feature_():
        """
        array feature of counting e.g. '.', '?', '!', ',' in a tweet (punctuation passed as 'key' parameter
        :return:
        """
        d_tw = load_twitter_data()
        tr_x_texts = branchify_twitter_extract_feature_loop(d_tw['train'], 'text')
        dv_x_texts = branchify_twitter_extract_feature_loop(d_tw['dev'], 'text')

        for i in range(len(tr_x_texts)):
            punctuations_counter = Counter(re.findall("[^\w\s]+", tr_x_texts[i]))
            punctuations = punctuations_counter[key]
            tr_x_texts[i] = punctuations

        for i in range(len(dv_x_texts)):
            punctuations_counter = Counter(re.findall("[^\w\s]+", dv_x_texts[i]))
            punctuations = punctuations_counter[key]
            dv_x_texts[i] = punctuations

        tr_x_texts, dv_x_texts = scale(tr_x_texts, dv_x_texts)

        return tr_x_texts, None, dv_x_texts
    return twitter_punctuation_count_feature_


def twitter_punctuation_contains_feature(key='!'):
    def twitter_punctuation_contains_feature_():
        """
        array feature of whether a tweet contains e.g. '.', '?', '!', ',' (punctuation passed as 'key' parameter
        :return:
        """
        d_tw = load_twitter_data()
        tr_x_texts = branchify_twitter_extract_feature_loop(d_tw['train'], 'text')
        dv_x_texts = branchify_twitter_extract_feature_loop(d_tw['dev'], 'text')

        for i in range(len(tr_x_texts)):
            punctuations_counter = Counter(re.findall("[^\w\s]+", tr_x_texts[i]))
            punctuations = 1 if punctuations_counter[key] > 0 else 0
            tr_x_texts[i] = punctuations

        for i in range(len(dv_x_texts)):
            punctuations_counter = Counter(re.findall("[^\w\s]+", dv_x_texts[i]))
            punctuations = 1 if punctuations_counter[key] > 0 else 0
            dv_x_texts[i] = punctuations

        tr_x_texts, dv_x_texts = scale(tr_x_texts, dv_x_texts)

        return tr_x_texts, None, dv_x_texts
    return twitter_punctuation_contains_feature_


def twitter_previous_tweet_similarity_feature(x_train, x_test, y_train, y_test, embeddings_index, source_similarity=0.5):
    def twitter_previous_tweet_similarity_feature_():
        """
        compute similarity between curent tweet (x_train / x_test) and the tweet before it (tr_x_previous /
        dv_x_previous). Source tweets have similarity set to an arbitrary value (e.g. 1,0,0.5)
        :return:
        """
        d_tw = load_twitter_data()
        tr_x_previous = branchify_twitter_extract_feature_from_previous_tweet_loop(d_tw['train'], 'text')
        dv_x_previous = branchify_twitter_extract_feature_from_previous_tweet_loop(d_tw['dev'], 'text')

        tr_x_previous = transform_description(tr_x_previous, embeddings_index)
        dv_x_previous = transform_description(dv_x_previous, embeddings_index)

        tr_x_previous = compare_embeddings_cosine(x_train, y_train, tr_x_previous, source_similarity)
        dv_x_previous = compare_embeddings_cosine(x_test, y_test, dv_x_previous, source_similarity)

        tr_x_previous, dv_x_previous = scale(tr_x_previous, dv_x_previous)

        return tr_x_previous, None, dv_x_previous

    return twitter_previous_tweet_similarity_feature_


def branchify_twitter_extract_feature_from_previous_tweet_loop(data, new_feature='text'):
    """Extract features from previous element of the current element in branches"""
    branches_previous_features = []
    for source_new_feature in data:
        ids_of_branches = source_new_feature['branches']  # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            next_to_add = ""

            for id in branch_ids:
                if source_new_feature['source']['id_str'] == id:  # if the id in question is the source post
                    if source_new_feature['source']['id_str'] != str(source_new_feature['source']['id']):
                        print(source_new_feature['source']['id_str'], source_new_feature['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_new_feature['source']['id_str'] != source_new_feature['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branches_previous_features.append(next_to_add)
                    next_to_add = source_new_feature['source'][new_feature]

                for reply in source_new_feature['replies']:  # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)

                        branches_previous_features.append(next_to_add)
                        next_to_add = reply[new_feature]

    return branches_previous_features


def compare_embeddings_cosine(orig_data_branches, orig_data_y, previous_emb, source_similarity):
    """
    :param orig_data_branches: original data organised into branches
    :param orig_data_y: y values of the original branches data to ignore zero-padded timesteps
    :param previous_emb: list of embeddings coresponding to previous tweet of the current one at orig_data_branches. If
    source, than emb. is np.zeros(EMBEDDING_DIM)
    :return: array/list of cosine similarities between current and embedding at the previous timestep
    """

    similarity_values = []

    tweet_counter = 0
    for i in range(len(orig_data_branches)):

        for j in range(len(orig_data_branches[i])):
            if not np.array_equal(orig_data_y[i][j], np.zeros(NUMBER_OF_CLASSES)):
                if j == 0:                                                  # if branch source
                    similarity_values.append(source_similarity)
                else:
                    similarity_values.append(1 - spatial.distance.cosine(orig_data_branches[i][j], np.array([previous_emb[tweet_counter]])))
                tweet_counter += 1

            else:
                similarity_values.append(0)

    similarity_values = np.asarray(similarity_values)

    return similarity_values