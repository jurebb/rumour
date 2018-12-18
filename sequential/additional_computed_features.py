from sequential.prepare_seq_data import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
from sequential.additional_features import *
import re
from html.parser import HTMLParser
from collections import Counter


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

