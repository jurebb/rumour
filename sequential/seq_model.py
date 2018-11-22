import sequential.prepare_seq_data as data
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, classification_report, \
                            make_scorer
from sklearn.model_selection import train_test_split
import numpy as np

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.layers import Dense, Activation, LSTM, Embedding, Dropout, Input, TimeDistributed
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.externals import joblib
from keras import backend as K

import pickle

_DATA_DIR = "/home/interferon/PycharmProjects/rumoureval19/rumour"      # directory where twitter.pkl and reddit.pkl
                                                                        # are located
GLOVE_DIR = "/home/interferon/Documents/dipl_projekt/glove/glove.twitter.27B.200d.txt"

### params
_NUM_CLASSES = 0
EMBEDDING_DIM = 200
MAX_SEQUENCE_LENGTH = 50
MAX_NB_WORDS = 200000

# comments and TODO:
# - tolower() prije tokeniziranja (i pretprocessiranje, opcenito - kidanje, stemmanje, micanje interp znakova itd???)
# - maskanje paddanih dijelova brancheva
# - dense 4 ili 5 neurona (num_classes ili num_classes+1? na y padam [0000] ili neku drugu klasu ili cak 5-dim vektor?


def class_to_onehot(y):
    y_oh = np.zeros((len(y), _NUM_CLASSES))
    y_oh[range(len(y)), y] = 1

    return y_oh


def lstm_model(max_branch_length, EMBEDDING_DIM, len_dataset):
    '''
    Describes and builds a keras LSTM model
    :return: the compiled model
    '''

    # TODO masking
    # TODO Dropout?

    model = Sequential()
    model.add(LSTM(128, dropout=0.1, recurrent_dropout=0.1, return_sequences=True))                       # , input_shape=(max_branch_length, EMBEDDING_DIM)
    model.add(Activation('sigmoid'))
    model.add(TimeDistributed(Dense(_NUM_CLASSES, activation='softmax')))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


def avg_embedding(branch, embedding_matrix, EMBEDDING_DIM, max_branch_length):
    """
    making an avg embedding vector (sum of word embeddings / num of words) for each tweet/reddit post
    it is also padded with zero vectors at the beginning so that all branches are the same length
    """
    new_branch = []

    len_zeros = max_branch_length - len(branch)

    if len_zeros > 0:
        for i in range(len_zeros):
            new_branch.append(np.zeros(EMBEDDING_DIM))

    for post in branch:         # post = tweet/reddit post
        sum_vector = np.zeros(EMBEDDING_DIM)
        word_count = 0
        for word_num in post:
            sum_vector = np.add(sum_vector, embedding_matrix[word_num])
            word_count += 1

        if word_count > 0:
            try:
                sum_vector = np.true_divide(sum_vector, word_count)
            except RuntimeWarning as e:
                print(e)
                print(word_count)
                print(np.isfinite(sum_vector).all())
                exit(0)

        new_branch.append(sum_vector)

    return new_branch


def pad_ys(y_branch, max_branch_length):
    new_y_branch = []

    len_zeros = max_branch_length - len(y_branch)

    if len_zeros > 0:
        for i in range(len_zeros):
            new_y_branch.append(np.zeros(_NUM_CLASSES))

    for i in range(len(y_branch)):
        new_y_branch.append(y_branch[i])

    return np.asarray(new_y_branch)


def make_embedding_matrix(word_index):
    """makes embedding matrix from GLOVE_DIR file"""
    emb_index = {}
    f = open(GLOVE_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            emb_index[word] = coefs
        except:
            pass
    f.close()

    print('Found %s word vectors.' % len(emb_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = emb_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print('embedding_matrix.shape', embedding_matrix.shape)

    return embedding_matrix


def onehot_encode_ys(tr_y, ts_y, dv_y):
    le = LabelEncoder()
    le = le.fit(list(y_values))

    tr_y = np.asarray(tr_y)
    for row_index in range(tr_y.shape[0]):
        tr_y[row_index] = np.asarray(class_to_onehot(le.transform(tr_y[row_index])))

    ts_y = np.asarray(ts_y)
    for row_index in range(ts_y.shape[0]):
        ts_y[row_index] = np.asarray(class_to_onehot(le.transform(ts_y[row_index])))

    dv_y = np.asarray(dv_y)
    for row_index in range(dv_y.shape[0]):
        dv_y[row_index] = np.asarray(class_to_onehot(le.transform(dv_y[row_index])))

    return le, tr_y, ts_y, dv_y


def convert_pred_to_onehot(label_encoder, y_pred):

    for row_index in range(y_pred.shape[0]):
        y_pred[row_index] = np.asarray(class_to_onehot(label_encoder.transform(y_pred[row_index])))

    return y_pred


def tokenize_datasets(x_values, tr_x, ts_x, dv_x):
    """tokenizes the datasets e.g. ->   'People of Reddit! Game of Thrones Bosses Confirm That Seasons 7 and 8 Will
                                            Be Shorter Than Ever Before, is it true?'
                                        ->
                                        [32, 4, 226, 1059, 4, 4756, 4757, 1381, 7, 2225, 1005, 5, 992, 44, 15,
                                            2242, 66, 318, 140, 6, 9, 91]
    """
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(list(x_values))

    train_sequences = []
    for row_index in range(tr_x.shape[0]):
        train_sequences.append(tokenizer.texts_to_sequences(tr_x[row_index]))

    test_sequences = []
    for row_index in range(ts_x.shape[0]):
        test_sequences.append(tokenizer.texts_to_sequences(ts_x[row_index]))

    dev_sequences = []
    for row_index in range(dv_x.shape[0]):
        dev_sequences.append(tokenizer.texts_to_sequences(dv_x[row_index]))

    return tokenizer, train_sequences, test_sequences, dev_sequences


def avg_emb_and_pad(max_branch_length, embedding_matrix, train_sequences, test_sequences, dev_sequences):
    """apply avg_embedding() function (avg_emb + pad) to train, test and dev sequences,
    e.g.
    [[32, 4, 230, ... 146, 6, 9, 92], [190, 57,...  4, 37, 4254, 2129], [1599, 3367]]   ... branch consisting of 3 posts
                                                                                            (e.g. tweets)
    ->
    np.array of dimension (max_branch_len, EMBEDDING_DIM), with np.zeros(EMBEDDING_DIM) vectors padded to the beginning,
    and avg embedding vector representation of each post
    """

    branches_train = []
    for branch in train_sequences:
        branches_train.append(avg_embedding(branch, embedding_matrix, EMBEDDING_DIM, max_branch_length))

    for branch_index in range(len(branches_train)):
        branches_train[branch_index] = np.asarray(branches_train[branch_index])

    branches_test = []
    for branch in test_sequences:
        branches_test.append(avg_embedding(branch, embedding_matrix, EMBEDDING_DIM, max_branch_length))

    for branch_index in range(len(branches_test)):
        branches_test[branch_index] = np.asarray(branches_test[branch_index])

    branches_dev = []
    for branch in dev_sequences:
        branches_dev.append(avg_embedding(branch, embedding_matrix, EMBEDDING_DIM, max_branch_length))

    for branch_index in range(len(branches_dev)):
        branches_dev[branch_index] = np.asarray(branches_dev[branch_index])

    return np.array(branches_train), np.array(branches_test), np.array(branches_dev)


def pad_y_datasets(max_branch_length, tr_y, ts_y, dv_y):
    """pad (max_branch_length - current_branch_length) [0,0,0...0] values to the beginning of each y branch"""

    for j in range(len(tr_y)):
        tr_y[j] = pad_ys(tr_y[j], max_branch_length)
    tr_y = list(tr_y)
    tr_y = np.array(tr_y)

    for j in range(len(ts_y)):
        ts_y[j] = pad_ys(ts_y[j], max_branch_length)
    ts_y = list(ts_y)
    ts_y = np.array(ts_y)

    for j in range(len(dv_y)):
        dv_y[j] = pad_ys(dv_y[j], max_branch_length)
    dv_y = list(dv_y)
    dv_y = np.array(dv_y)

    return tr_y, ts_y, dv_y


if __name__ == "__main__":

    d_tw = data.load_reddit_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = data.branchify_data(d_tw, data.branchify_reddit_extraction_loop)

    #################################################################################################################
    tr_x = np.asarray(tr_x)
    ts_x = np.asarray(ts_x)
    dv_x = np.asarray(dv_x)

    # store all possible y values
    y_values = set()
    for row_index in range(len(tr_y)):
        y_values.update(tr_y[row_index])

    _NUM_CLASSES = len(y_values)

    le, tr_y, ts_y, dv_y = onehot_encode_ys(tr_y, ts_y, dv_y)

    # store all possible x values
    x_values = set()
    for row_index in range(len(tr_x)):
        x_values.update(tr_x[row_index])

    tokenizer, train_sequences, test_sequences, dev_sequences = tokenize_datasets(x_values, tr_x, ts_x, dv_x)

    word_index = tokenizer.word_index

    embedding_matrix = make_embedding_matrix(word_index)

    # making an avg embedding vector (sum of word embeddings / num of words) for each tweet/reddit post
    max_branch_length = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))

    branches_train, branches_test, branches_dev = avg_emb_and_pad(max_branch_length, embedding_matrix,
                                                                  train_sequences, test_sequences, dev_sequences)

    # pad the ys the same way
    tr_y, ts_y, dv_y = pad_y_datasets(max_branch_length, tr_y, ts_y, dv_y)
    ##################################################################################################################

    model = lstm_model(max_branch_length, EMBEDDING_DIM, len(branches_train))

    model.fit(branches_train, tr_y, batch_size=64, epochs=8)

    score = model.evaluate(branches_dev, dv_y, batch_size=128)
    print('model.evaluate', score)

    y_pred = model.predict(branches_dev)
    # y_pred = convert_pred_to_onehot(le, y_pred)
    print('classification report:\n', classification_report(dv_y, y_pred))