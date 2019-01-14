import pandas as pd
import os

_DATA_DIR = "C:\\Users\\viktor\\Projects\\Python\\data_set\\data"      # directory where twitter.pkl and reddit.pkl are located
#  _DATA_DIR ="/home/interferon/PycharmProjects/rumoureval19/rumour"


def load_twitter_data():
    """Loads twitter dataset in twitter.pkl"""

    os.chdir(_DATA_DIR)
    data = pd.read_pickle('twitter_new2.pkl')

    return data


def load_reddit_data():
    """Loads reddit dataset in reddit.pkl"""

    os.chdir(_DATA_DIR)
    data = pd.read_pickle('reddit_new2.pkl')

    return data


def sourcify_data(data, branchify_function):
    """Extract source posts from loaded dataset"""

    # train
    train_sources_texts = []
    train_sources_veracities = []
    branchify_function(data['train'], train_sources_texts, train_sources_veracities)

    # test
    test_sources_texts = []
    test_sources_veracities = []
    branchify_function(data['test'], test_sources_texts, test_sources_veracities)

    # dev
    dev_sources_texts = []
    dev_sources_veracities = []
    branchify_function(data['dev'], dev_sources_texts, dev_sources_veracities)

    return train_sources_texts, train_sources_veracities, test_sources_texts, test_sources_veracities, \
            dev_sources_texts, dev_sources_veracities


def source_tweet_extraction_loop(data, sources_texts, sources_veracities):
    """Extract source tweets from twitter data"""

    for source_text in data:

        sources_texts.append(source_text['source']['text'])
        sources_veracities.append(source_text['veracity'])


def source_reddit_extraction_loop(data, sources_texts, sources_veracities):
    """Extract source reddit posts from reddit data"""

    for source_text in data:

        sources_texts.append(source_text['source']['text'])
        sources_veracities.append(source_text['veracity'])


def print_structure_example():
    """Print dataset structure examples for both datasets"""

    pass


def branchify_twitter_taskb_extraction_loop(data, branches_texts, branches_labels):
    """Extract tweets from ids of branches"""

    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            branch_texts = []

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != str(source_text['source']['id']):
                        print(source_text['source']['id_str'], source_text['source']['id'])
                        raise AssertionError("twitter source id_str and source id don't match for ", id)
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("twitter source id_str and id don't match for ", id)

                    branch_texts.append(source_text['source']['text'])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if reply['id_str'] != str(reply['id']):
                            raise AssertionError("twitter reply id_str and id don't match for ", id)

                        branch_texts.append(reply['text'])

            branches_texts.append(branch_texts)
            branches_labels.append(source_text['veracity'])


def branchify_reddit_taskb_extraction_loop(data, branches_texts, branches_labels):
    """Extract reddit posts from ids of branches"""
    za_ispisati = True
    for source_text in data:
        ids_of_branches = source_text['branches']   # gives a list of branches ids from json structure
        for branch_ids in ids_of_branches:
            branch_texts = []

            for id in branch_ids:
                if source_text['source']['id_str'] == id:       # if the id in question is the source post
                    if source_text['source']['id_str'] != source_text['id']:
                        raise AssertionError("reddit source id_str and id don't match for ", id)

                    branch_texts.append(source_text['source']['text'])

                for reply in source_text['replies']:            # if the id in question is the reply of the source post
                    if reply['id_str'] == id:
                        if za_ispisati:
                            print(reply.keys())
                            print('DATA user_reports')
                            print(reply['data']['user_reports'])
                            print()
                            print('id_str')
                            print(reply['id_str'])
                            za_ispisati = False
                        branch_texts.append(reply['text'])

            branches_texts.append(branch_texts)
            branches_labels.append(source_text['veracity'])


if __name__ == "__main__":

    print_structure_example()

    d_tw = load_twitter_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = sourcify_data(d_tw, source_tweet_extraction_loop)
    print('len(tr_x)', len(tr_x))
    print('len(tr_y)', len(tr_y))
    print('len(ts_x)', len(ts_x))
    print('len(ts_y)', len(ts_y))
    print('len(dv_x)', len(dv_x))
    print('len(dv_y)', len(dv_y))

    print('==================================================')

    d_rd = load_reddit_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = sourcify_data(d_rd, source_reddit_extraction_loop)
    print('len(tr_x)', len(tr_x))
    print('len(tr_y)', len(tr_y))
    print('len(ts_x)', len(ts_x))
    print('len(ts_y)', len(ts_y))
    print('len(dv_x)', len(dv_x))
    print('len(dv_y)', len(dv_y))
