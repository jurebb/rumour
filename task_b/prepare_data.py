import pandas as pd
import os

# _DATA_DIR = "C:\\Users\\viktor\\Projects\\Python\\data_set\\data"      # directory where twitter.pkl and reddit.pkl are located
_DATA_DIR ="/home/interferon/PycharmProjects/rumoureval19/rumour"


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
