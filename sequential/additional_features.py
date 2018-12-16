from sequential.prepare_seq_data import *
from sklearn.preprocessing import StandardScaler
import numpy as np

def twitter_length():
    """
    length of tweets in words
    :return:
    """
    pass


def scale(x_train, x_test):

    x_train = np.asarray(x_train).reshape((-1, 1))
    x_test = np.asarray(x_test).reshape((-1, 1))
    ss = StandardScaler()
    x_train = ss.fit_transform(x_train)
    x_test = ss.transform(x_test)
    x_train = np.asarray(x_train).reshape(-1)
    x_test = np.asarray(x_test).reshape(-1)

    return x_train, x_test


def twitter_user_id_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_username_ids = twitter_user_id_extraction_loop(d_tw['train'])
    dv_x_username_ids = twitter_user_id_extraction_loop(d_tw['dev'])

    tr_x_username_ids, dv_x_username_ids = scale(tr_x_username_ids, dv_x_username_ids)

    '''
    maximum_id = max(tr_x_username_ids)
    minimum_id = min(tr_x_username_ids)

    for i in range(len(tr_x_username_ids)):
        for j in range(len(tr_x_username_ids[i])):
            tr_x_username_ids[i][j] = (tr_x_username_ids[i][j] - minimum_id) / (maximum_id - minimum_id)

    for i in range(len(dv_x_username_ids)):
        for j in range(len(dv_x_username_ids[i])):
            dv_x_username_ids[i][j] = (dv_x_username_ids[i][j] - minimum_id) / (maximum_id - minimum_id)
    '''
    return tr_x_username_ids, None, dv_x_username_ids

def twitter_retweet_count_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_retweet_count = branchify_twitter_extract_feature_loop(d_tw['train'], 'retweet_count')
    dv_x_retweet_count = branchify_twitter_extract_feature_loop(d_tw['dev'], 'retweet_count')

    tr_x_retweet_count, dv_x_retweet_count = scale(tr_x_retweet_count, dv_x_retweet_count)

    return tr_x_retweet_count, None, dv_x_retweet_count

def twitter_favourited_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_favourited = branchify_twitter_extract_feature_loop(d_tw['train'], 'favourited')
    dv_x_favourited = branchify_twitter_extract_feature_loop(d_tw['dev'], 'favourited')

    return tr_x_favourited, None, dv_x_favourited

def twitter_profile_use_background_image_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_profile_use_background_image = branchify_twitter_extract_user_feature_loop(d_tw['train'], 'profile_use_background_image')
    dv_x_profile_use_background_image = branchify_twitter_extract_user_feature_loop(d_tw['dev'], 'profile_use_background_image')

    return tr_x_profile_use_background_image, None, dv_x_profile_use_background_image

def twitter_profile_favourites_count():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_profile_favourites_count = branchify_twitter_extract_user_feature_loop(d_tw['train'], 'favourites_count')
    dv_x_profile_favourites_count = branchify_twitter_extract_user_feature_loop(d_tw['dev'], 'favourites_count')

    tr_x_profile_favourites_count, dv_x_profile_favourites_count = scale(tr_x_profile_favourites_count, dv_x_profile_favourites_count)

    return tr_x_profile_favourites_count, None, dv_x_profile_favourites_count


def branchify_twitter_extract_feature_loop(data, new_feature='retweet_count'):
    """Extract features from ids of branches"""
    branches_new_features = []
    for source_new_feature in data:
        ids_of_branches = source_new_feature['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            #branch_new_features = []

            for id in branch_ids:
                if source_new_feature['source']['id_str'] == id:       # if the id in question is the source post
                    if source_new_feature['source']['id_str'] != str(source_new_feature['source']['id']):
                        print(source_new_feature['source']['id_str'], source_new_feature['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_new_feature['source']['id_str'] != source_new_feature['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branches_new_features.append(source_new_feature['source'][new_feature])

                for reply in source_new_feature['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)
                
                        branches_new_features.append(reply[new_feature])

            #branches_new_features.append(branch_new_features)

    return branches_new_features


def branchify_twitter_extract_user_feature_loop(data, new_feature='profile_use_background_image'):
    """Extract features from ids of branches"""
    branches_new_features = []
    for source_new_feature in data:
        ids_of_branches = source_new_feature['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            #branch_new_features = []

            for id in branch_ids:
                if source_new_feature['source']['id_str'] == id:       # if the id in question is the source post
                    if source_new_feature['source']['id_str'] != str(source_new_feature['source']['id']):
                        print(source_new_feature['source']['id_str'], source_new_feature['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_new_feature['source']['id_str'] != source_new_feature['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branches_new_features.append(source_new_feature['source']['user'][new_feature])

                for reply in source_new_feature['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)
                
                        branches_new_features.append(reply['user'][new_feature])

            #branches_new_features.append(branch_new_features)

    return branches_new_features


def branchify_twitter_extract_time_loop(data):
    """Extract features from ids of branches"""
    ispisi = False
    branches_new_features = []
    for source_new_feature in data:
        ids_of_branches = source_new_feature['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            #branch_new_features = []

            for id in branch_ids:
                if source_new_feature['source']['id_str'] == id:       # if the id in question is the source post
                    if source_new_feature['source']['id_str'] != str(source_new_feature['source']['id']):
                        print(source_new_feature['source']['id_str'], source_new_feature['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_new_feature['source']['id_str'] != source_new_feature['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branches_new_features.append(0)

                for reply in source_new_feature['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)

                        time1 = reply['created_at'].split()
                        time2 = source_new_feature['source']['created_at'].split()
                        if time1[0] == time2[0] and time1[1] == time2[1] and time1[2] == time2[2]:
                            time1 = time1[3].split(':')
                            for k in range(len(time1)):
                                if time1[k][0] == '0':
                                    time1[k] = time1[k][1]
                            time1 = eval(time1[0]) + eval(time1[1]) / 60  + eval(time1[2]) / 3600

                            time2 = time2[3].split(':')
                            for k in range(len(time2)):
                                if time2[k][0] == '0':
                                    time2[k] = time2[k][1]
                            time2 = eval(time2[0]) + eval(time2[1]) / 60  + eval(time2[2]) / 3600
                            
                            branches_new_features.append(time1 - time2) # - source_new_feature['source']['created_at'])
                        else:
                            branches_new_features.append(24)
                        
                        if ispisi:
                            print(branch_new_features)
                            print(reply['created_at'])
                            print(source_new_feature['source']['created_at'])
                            ispisi = False

            #branches_new_features.append(branch_new_features)

    return branches_new_features



def twitter_user_id_extraction_loop(data):
    branches_usernames = []

    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != str(source_text['source']['id']):
                        print(source_text['source']['id_str'], source_text['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branches_usernames.append(source_text['source']['user']['id'])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)

                        branches_usernames.append(reply['user']['id'])
    return branches_usernames