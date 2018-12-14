from sequential.prepare_seq_data import *


def twitter_length():
    """
    length of tweets in words
    :return:
    """
    pass


def twitter_user_id_feature():
    """
    extract user ids the same way text is extracted to be concatenated
    :return:
    """
    d_tw = load_twitter_data()
    tr_x_username_ids, _, ts_x_username_ids, _, dv_x_username_ids, _ = branchify_data(d_tw, twitter_user_id_extraction_loop)

    return tr_x_username_ids, ts_x_username_ids, dv_x_username_ids


def twitter_user_id_extraction_loop(data, branches_usernames, branches_labels):
    unique_usernames = dict()

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
                    branches_labels.append(source_text['source']['label'])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)

                        branches_usernames.append(reply['user']['id'])
                        branches_labels.append(reply['label'])