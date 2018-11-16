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
from keras.layers import Dense, Activation, LSTM, Embedding, Dropout, Input
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
_NUM_CLASSES = 0

# comments and TODO:
# - tolower() prije tokeniziranja (i pretprocessiranje, opcenito - kidanje, stemmanje, micanje interp znakova itd???)


def class_to_onehot(y):
    y_oh = np.zeros((len(y), _NUM_CLASSES))
    y_oh[range(len(y)), y] = 1

    return y_oh


def lstm_model(max_branch_length):
    '''
    Describes and bhuilds a kersan LSTM model
    :return: the compiled model
    '''

    # TODO masking
    # TODO Dropout?

    model = Sequential()
    # model.add(Embedding(10, 32))
    model.add(LSTM(32))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
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


if __name__ == "__main__":

    d_tw = data.load_reddit_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = data.branchify_data(d_tw, data.branchify_reddit_extraction_loop)

    #################################################################################################################
    tr_x = np.asarray(tr_x)
    dv_x = np.asarray(dv_x)

    # store all possible y values
    y_values = set()
    for row_index in range(len(tr_y)):
        y_values.update(tr_y[row_index])

    _NUM_CLASSES = len(y_values)

    le = LabelEncoder()
    le = le.fit(list(y_values))

    tr_y = np.asarray(tr_y)
    for row_index in range(tr_y.shape[0]):
        tr_y[row_index] = np.asarray(class_to_onehot(le.transform(tr_y[row_index])))

    EMBEDDING_DIM = 200
    MAX_SEQUENCE_LENGTH = 50
    MAX_NB_WORDS = 200000

    # store all possible x values
    x_values = set()
    for row_index in range(len(tr_x)):
        x_values.update(tr_x[row_index])

    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(list(x_values))

    train_sequences = []
    for row_index in range(tr_x.shape[0]):
        train_sequences.append(tokenizer.texts_to_sequences(tr_x[row_index]))

    # for i in range(len(train_sequences)):
    #         train_sequences[i] = sequence.pad_sequences(train_sequences[i], maxlen=MAX_SEQUENCE_LENGTH)

    dev_sequences = []
    for row_index in range(dv_x.shape[0]):
        dev_sequences.append(tokenizer.texts_to_sequences(dv_x[row_index]))

    # for i in range(len(dev_sequences)):
    #     dev_sequences[i] = sequence.pad_sequences(dev_sequences[i], maxlen=MAX_SEQUENCE_LENGTH)

    word_index = tokenizer.word_index

    embeddings_index = {}
    f = open(GLOVE_DIR, encoding="utf8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except:
            pass
    f.close()

    print('Found %s word vectors.' % len(embeddings_index))

    embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    print(embedding_matrix.shape)

    # making an avg embedding vector (sum of word embeddings / num of words) for each tweet/reddit post
    max_branch_length = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))

    branches_train = []
    for branch in train_sequences:
        branches_train.append(avg_embedding(branch, embedding_matrix, EMBEDDING_DIM, max_branch_length))

    # pad the ys the same way
    for j in range(len(tr_y)):
        tr_y[j] = pad_ys(tr_y[j], max_branch_length)

    # sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    ##################################################################################################################

    # x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    # x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    # print('x_train shape:', x_train.shape)
    # print('x_test shape:', x_test.shape)

    model = lstm_model(max_branch_length)

    model.fit(branches_train, tr_y, batch_size=128, epochs=1)
    score = model.evaluate(ts_x, ts_y, batch_size=128)

    print('model.evaluate', score)

    y_pred = model.predict_classes(ts_x)

    print('classification report:\n', classification_report(ts_y, y_pred))