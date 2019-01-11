# model that uses information from task A model,
# e.g. ratio of SDQC labels, number of them, etc.

from sklearn.model_selection import KFold
from keras.layers import LSTM, Dense, Embedding, Input
from keras.models import Sequential
from keras.layers import Activation, TimeDistributed
from keras.optimizers import Adam

from sklearn.metrics import accuracy_score

from sequential.additional_computed_features import *
from sequential.feature_utils import *

from numpy.random import seed
from tensorflow import set_random_seed
seed(12)
set_random_seed(22)

MAX_BRANCH_LENGTH = -1
NUMBER_OF_CLASSES = 4


def kfold_feature():
    d_tw = load_twitter_data()
    tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_twitter_extraction_loop)

    # d_tw = load_reddit_data()
    # tr_x, tr_y, _, _, dv_x, dv_y = branchify_data(d_tw, branchify_reddit_extraction_loop)

    MAX_BRANCH_LENGTH = max(len(max(dv_x, key=len)), len(max(tr_x, key=len)))
    print('computed MAX_BRANCH_LENGTH =', MAX_BRANCH_LENGTH)

    embeddings_index = make_embeddings_index()

    x_train = transform_data(tr_x, embeddings_index, MAX_BRANCH_LENGTH)
    y_train = transform_labels(tr_y, MAX_BRANCH_LENGTH)

    x_test = transform_data(dv_x, embeddings_index, MAX_BRANCH_LENGTH)
    y_test = transform_labels(dv_y, MAX_BRANCH_LENGTH)

    #### feature engineering
    for new_feature_f in [twitter_favorite_count_feature, twitter_retweet_count_feature,
                          twitter_punctuation_count_feature('?'),
                          twitter_word_counter_feature, twitter_url_counter_feature,
                          twitter_previous_tweet_similarity_feature(x_train, x_test, y_train, y_test, embeddings_index),
                          twitter_user_mention_count_feature]:
        x_train, _, x_test = concat_features([new_feature_f], x_train, None, x_test, y_train, None, y_test)
        print('x_train.shape', x_train.shape)

    preds_train_meta = []

    kfold_iteration = 0
    kf = KFold(n_splits=5)

    for train_index, test_index in kf.split(x_train):
        x_train_base = x_train[train_index]
        x_test_base = x_train[test_index]
        y_train_base = y_train[train_index]
        y_test_base = y_train[test_index]

        print('>>> iteration:', kfold_iteration)

        print('x_train_base.shape', x_train_base.shape)
        print('x_test_base.shape', x_test_base.shape)
        print('y_train_base.shape', y_train_base.shape)
        print('y_test_base.shape', y_test_base.shape)

        model = Sequential()
        model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                       input_shape=(x_train.shape[1], x_train.shape[2])))
        model.add(Activation('sigmoid'))
        model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
        model.add(Activation('softmax'))
        adam = Adam(lr=0.001)
        model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
        model.fit(x_train_base, y_train_base, nb_epoch=8, batch_size=64)  # nb_epoch=50

        kfold_iteration += 1

        preds_test_base = model.predict(x_test_base, 100)
        preds_train_meta.append(preds_test_base)

    preds_train_meta = np.asarray(preds_train_meta)
    # preds_train_meta = np.reshape(preds_train_meta, (-1, preds_train_meta[0].shape[1], preds_train_meta[0].shape[2]))
    preds_train_meta = np.reshape(preds_train_meta, (-1, preds_train_meta[0].shape[2]))

    model = Sequential()
    model.add(LSTM(units=100, dropout=0.1, recurrent_dropout=0.1, return_sequences=True,
                   input_shape=(x_train.shape[1], x_train.shape[2])))
    model.add(Activation('sigmoid'))
    model.add(TimeDistributed(Dense(NUMBER_OF_CLASSES)))
    model.add(Activation('softmax'))
    adam = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    model.fit(x_train, y_train, nb_epoch=8, batch_size=64)  # nb_epoch=50

    preds_test_meta = model.predict(x_test, 100)
    preds_test_meta = np.reshape(preds_test_meta, (-1, preds_test_meta.shape[2]))

    preds_train_meta = [np.argmax(xx) for xx in preds_train_meta]
    preds_test_meta = [np.argmax(xx) for xx in preds_test_meta]

    preds_train_meta, preds_test_meta = scale(preds_train_meta, preds_test_meta)

    return preds_train_meta, None, preds_test_meta


if __name__ == "__main__":
    kfold_feature()
