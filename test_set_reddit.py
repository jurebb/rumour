import os
import json
from copy import deepcopy
import numpy as np
from copy import deepcopy
import pickle


def tree2branches(root):
    node = root
    parent_tracker = list()
    parent_tracker.append(root)
    branch = []
    branches = []
    i = 0
    siblings = None
    while True:
        node_name = list(node.keys())[i]
        branch.append(node_name)
        # get children of the node
        # actually all chldren, all tree left under this node
        first_child = list(node.values())[i]
        if first_child != []:  # if node has children
            node = first_child  # walk down
            parent_tracker.append(node)
            siblings = list(first_child.keys())
            i = 0  # index of a current node
        else:
            branches.append(deepcopy(branch))
            if siblings is not None:
                i = siblings.index(node_name)  # index of a current node
                # if node doesnt have next siblings
                while i+1 >= len(siblings):
                    if node is parent_tracker[0]:  # if it is a root node
                        return branches
                    del parent_tracker[-1]
                    del branch[-1]
                    node = parent_tracker[-1]  # walk up ... one step
                    node_name = branch[-1]
                    siblings = list(node.keys())
                    i = siblings.index(node_name)
                i = i+1    # ... walk right
    #            node =  parent_tracker[-1].values()[i]
                del branch[-1]
            else:
                return branches


def listdir_nohidden(path):
    contents = os.listdir(path)
    new_contents = [i for i in contents if i[0] != '.']

    return new_contents


def load_data():
    path = 'data\\data\\rumoureval-2019-test-data\\rumoureval-2019-test-data\\reddit-test-data'

    conversation_ids = listdir_nohidden(path)
    conversations = {}

    test = []

    for id in conversation_ids:
        conversation = {}
        conversation['id'] = id
        path_src = path + '/' + id + '/source-tweet'
        files_t = sorted(listdir_nohidden(path_src))
        with open(os.path.join(path_src, files_t[0])) as f:
            for line in f:
                src = json.loads(line)

                src['text'] = src['data']['children'][0]['data']['title']
                src['user'] = src['data']['children'][0]['data']['author']

                if files_t[0].endswith('.json'):
                    filename = files_t[0][:-5]
                    src['id_str'] = filename
                else:
                    print("No, no I don't like that")

                src['used'] = 0
                conversation['source'] = src

        tweets = []
        path_repl = path + '/' + id + '/replies'
        files_t = sorted(listdir_nohidden(path_repl))
        for repl_file in files_t:
            with open(os.path.join(path_repl, repl_file)) as f:
                for line in f:
                    tw = json.loads(line)

                    if 'body' in list(tw['data'].keys()):

                        tw['text'] = tw['data']['body']
                        tw['user'] = tw['data']['author']

                        if repl_file.endswith('.json'):
                            filename = repl_file[:-5]
                            tw['id_str'] = filename
                        else:
                            print("No, no I don't like that reply")

                        tw['used'] = 0

                        tweets.append(tw)
                    else:

                        tw['text'] = ''
                        tw['user'] = ''
                        tw['used'] = 0
                        if repl_file.endswith('.json'):
                            filename = repl_file[:-5]
                            tw['id_str'] = filename
                        else:
                            print("No, no I don't like that reply")

                        tweets.append(tw)

        conversation['replies'] = tweets
        path_struct = path + '/' + id + '/structure.json'

        with open(path_struct, 'r') as f:
            struct = json.load(f)
            conversation['structure'] = struct
            branches = tree2branches(conversation['structure'])
            conversation['branches'] = branches

        test.append(conversation)

    return test

if __name__ == "__main__":
    data = load_data()
    for i in range(len(data)):
        try:
            print(data[i])
        except:
            pass
    for dd in data:
        try:
            print(dd.keys())
        except:
            print('blin')
    
    new_data = dict()
    new_data['test'] = data
    with open('reddit_test.pkl', 'wb') as f:
        pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)
