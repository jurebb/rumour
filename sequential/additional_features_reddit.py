from sequential.prepare_seq_data import *
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
import numpy as np

def scale(x_train, x_test):

    x_train = np.asarray(x_train).reshape((-1, 1))
    x_test = np.asarray(x_test).reshape((-1, 1))
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    x_train = np.asarray(x_train).reshape(-1)
    x_test = np.asarray(x_test).reshape(-1)

    return x_train, x_test

def scale2(x_train, x_test):

    x_train = np.asarray(x_train).reshape((-1, 1))
    x_test = np.asarray(x_test).reshape((-1, 1))
    ss = MinMaxScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    x_train = np.asarray(x_train).reshape(-1)
    x_test = np.asarray(x_test).reshape(-1)

    return x_train, x_test

def reddit_kind_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_reddit_data()
    tr_x_kind = branchify_reddit_extract_feature_loop(d_tw['train'], 'kind')
    dv_x_kind = branchify_reddit_extract_feature_loop(d_tw['dev'], 'kind')

    lb = LabelBinarizer()
    tr_x_kind = lb.fit_transform(tr_x_kind)
    print(tr_x_kind.shape)
    print(tr_x_kind[0])
    dv_x_kind = lb.transform(dv_x_kind)

    return tr_x_kind, None, dv_x_kind

def reddit_id_str_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_reddit_data()
    tr_x_id_str = branchify_reddit_extract_feature_loop(d_tw['train'], 'id_str')
    dv_x_id_str = branchify_reddit_extract_feature_loop(d_tw['dev'], 'id_str')

    lb = LabelBinarizer()
    tr_x_id_str = np.argmax(lb.fit_transform(tr_x_id_str), axis=1)
    dv_x_id_str = np.argmax(lb.transform(dv_x_id_str), axis=1)

    print(tr_x_id_str.shape)
    print(dv_x_id_str[0])

    return tr_x_id_str, None, dv_x_id_str

def reddit_used_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_reddit_data()
    tr_x_used = branchify_reddit_extract_feature_loop(d_tw['train'], 'used')
    dv_x_used = branchify_reddit_extract_feature_loop(d_tw['dev'], 'used')

    tr_x_used, dv_x_used = scale(tr_x_used, dv_x_used)

    return tr_x_used, None, dv_x_used

def reddit_likes_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_reddit_data()
    tr_x_likes = branchify_reddit_extract_feature_from_data_loop(d_tw['train'], 'likes')
    dv_x_likes = branchify_reddit_extract_feature_from_data_loop(d_tw['dev'], 'likes')

    tr_x_likes, dv_x_likes = scale(tr_x_likes, dv_x_likes)

    return tr_x_likes, None, dv_x_likes

def reddit_score_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_reddit_data()
    tr_x_score = branchify_reddit_extract_feature_from_data_loop(d_tw['train'], 'score')
    dv_x_score = branchify_reddit_extract_feature_from_data_loop(d_tw['dev'], 'score')

    tr_x_score, dv_x_score = scale(tr_x_score, dv_x_score)

    return tr_x_score, None, dv_x_score

def reddit_controversiality_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_reddit_data()
    tr_x_controversiality = branchify_reddit_extract_controversiality_loop(d_tw['train'])
    dv_x_controversiality = branchify_reddit_extract_controversiality_loop(d_tw['dev'])

    tr_x_controversiality, dv_x_controversiality = scale(tr_x_controversiality, dv_x_controversiality)

    return tr_x_controversiality, None, dv_x_controversiality


def branchify_reddit_extract_feature_loop(data, new_feature='kind'):
    """Extract reddit posts from ids of branches"""
    za_ispisati = True
    branches_new_feature = []
    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branches_new_feature.append(source_text['source'][new_feature])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        branches_new_feature.append(reply[new_feature])
    return branches_new_feature


def branchify_reddit_extract_feature_from_data_loop(data, new_feature='likes'):
    """Extract reddit posts from ids of branches"""
    za_ispisati = True
    branches_new_feature = []
    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)
                    branches_new_feature.append(source_text['source']['data']['children'][0]['data'][new_feature])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        try:
                            branches_new_feature.append(reply['data'][new_feature])
                        except KeyError:
                            branches_new_feature.append(0) #SUMNJIVO
    return branches_new_feature


def branchify_reddit_extract_controversiality_loop(data):
    """Extract reddit posts from ids of branches"""
    za_ispisati = True
    branches_new_feature = []
    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)
                    branches_new_feature.append(-1) #SUMNJIVO

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        try:
                            branches_new_feature.append(reply['data']['controversiality'])
                        except KeyError:
                            branches_new_feature.append(0) #SUMNJIVO
    return branches_new_feature


