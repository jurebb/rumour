import os
import json
from data_set_reddit import tree2branches
import pickle

# %%
def load_dataset():
    # Load labels and split for task A and task B
    # Load folds and conversations
    path_to_folds = 'data\\data\\rumoureval-2019-test-data\\rumoureval-2019-test-data\\twitter-en-test-data'
    folds = sorted(os.listdir(path_to_folds))
    newfolds = [i for i in folds if i[0] != '.']
    folds = newfolds
    cvfolds = {}
    allconv = []
    test = []
    for nfold, fold in enumerate(folds):
        path_to_tweets = os.path.join(path_to_folds, fold)
        tweet_data = sorted(os.listdir(path_to_tweets))
        newfolds = [i for i in tweet_data if i[0] != '.']
        tweet_data = newfolds
        conversation = {}
        for foldr in tweet_data:
            flag = 0
            conversation['id'] = foldr
            tweets = []
            path_repl = path_to_tweets + '/' + foldr + '/replies'
            files_t = sorted(os.listdir(path_repl))
            newfolds = [i for i in files_t if i[0] != '.']
            files_t = newfolds
            if files_t != []:
                for repl_file in files_t:
                    with open(os.path.join(path_repl, repl_file)) as f:
                        for line in f:
                            tw = json.loads(line)
                            tw['used'] = 0
                            replyid = tw['id_str']
                            tweets.append(tw)
                            if tw['text'] is None:
                                print("Tweet has no text", tw['id'])
                conversation['replies'] = tweets

                path_src = path_to_tweets + '/' + foldr + '/source-tweet'
                files_t = sorted(os.listdir(path_src))
                with open(os.path.join(path_src, files_t[0])) as f:
                    for line in f:
                        src = json.loads(line)
                        src['used'] = 0
                        scrcid = src['id_str']
                        src['set'] = flag

                conversation['source'] = src
                if src['text'] is None:
                    print("Tweet has no text", src['id'])
                path_struct = path_to_tweets + '/' + foldr + '/structure.json'
                with open(path_struct) as f:
                    for line in f:
                        struct = json.loads(line)
                if len(struct) > 1:
                    # I had to alter the structure of this conversation
                    if foldr == '553480082996879360':
                        new_struct = {}
                        new_struct[foldr] = struct[foldr]
                        new_struct[foldr]['553495625527209985'] = struct['553485679129534464']['553495625527209985']
                        new_struct[foldr]['553495937432432640'] = struct['553490097623269376']['553495937432432640']
                        struct = new_struct
                    else:
                        new_struct = {}
                        new_struct[foldr] = struct[foldr]
                        struct = new_struct
                    # Take item from structure if key is same as source tweet id
                conversation['structure'] = struct

                branches = tree2branches(conversation['structure'])
                conversation['branches'] = branches
                test.append(conversation.copy())
                allconv.append(conversation.copy())
            
        cvfolds[fold] = allconv
        allconv = []

    return test

if __name__ == "__main__":
    data = load_dataset()
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
    with open('twitter_test.pkl', 'wb') as f:
        pickle.dump(new_data, f, pickle.HIGHEST_PROTOCOL)