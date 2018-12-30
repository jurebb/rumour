import pandas as pd
import os

# _DATA_DIR = "C:\\Users\\viktor\\Projects\\Python\\data_set\\data"      # directory where twitter.pkl and reddit.pkl are located
_DATA_DIR ="/home/interferon/PycharmProjects/rumoureval19/rumour"
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
    data = pd.read_pickle('twitter_new2.pkl')

    return data


def load_reddit_data():
    """Loads reddit dataset in reddit.pkl"""

    os.chdir(_DATA_DIR)
    data = pd.read_pickle('reddit_new2.pkl')

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
    za_ispisati = True
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
                        if za_ispisati:
                            print(reply.keys())
                            print('DATA user_reports')
                            print(reply['data']['user_reports'])
                            print()
                            print('id_str')
                            print(reply['id_str'])
                            za_ispisati = False
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


if __name__ == "__main__":

    print_structure_example()

    d_tw = load_twitter_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    print('len(tr_x)', len(tr_x))
    print('tr_x[0:10]', tr_x[0:10])
    print('len(tr_y)', len(tr_y))
    print('tr_y[0:10]', tr_y[0:10])
    print('len(ts_x)', len(ts_x))
    print('len(ts_y)', len(ts_y))
    print('len(dv_x)', len(dv_x))
    print('dv_x[0:10]', dv_x[0:10])
    print('len(dv_y)', len(dv_y))
    print('dv_y[0:10]', dv_y[0:10])

    print('==================================================')

    d_rd = load_reddit_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = branchify_data(d_rd, branchify_reddit_extraction_loop)

    print('len(tr_x)', len(tr_x))
    print('tr_x[0:9]')
    for branch in range(0,9):
        print(tr_x[branch])
        print(tr_y[branch])
    print('len(tr_y)', len(tr_y))
    print('tr_y[0:10]', tr_y[0:10])
    print('len(ts_x)', len(ts_x))
    print('len(ts_y)', len(ts_y))
    print('len(dv_x)', len(dv_x))
    print('dv_x[0:10]', dv_x[0:10])
    print('len(dv_y)', len(dv_y))
    print('dv_y[0:10]', dv_y[0:10])
