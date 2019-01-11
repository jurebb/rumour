from task_b.prepare_data import *
from sequential.feature_utils import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer


def baseline_model_emb():
    d_tw = load_twitter_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = sourcify_data(d_tw, source_tweet_extraction_loop)

    d_rd = load_reddit_data()
    rtr_x, rtr_y, rts_x, rts_y, rdv_x, rdv_y = sourcify_data(d_rd, source_reddit_extraction_loop)

    embeddings_index = make_embeddings_index()


def baseline_model_tfidf():
    d_tw = load_twitter_data()
    tr_x, tr_y, ts_x, ts_y, dv_x, dv_y = sourcify_data(d_tw, source_tweet_extraction_loop)

    d_rd = load_reddit_data()
    rtr_x, rtr_y, rts_x, rts_y, rdv_x, rdv_y = sourcify_data(d_rd, source_reddit_extraction_loop)

    x_train = np.concatenate((tr_x, rtr_x), axis=0)
    x_test = np.concatenate((dv_x, rdv_x), axis=0)
    y_train = np.concatenate((tr_y, rtr_y), axis=0)
    y_test = np.concatenate((dv_y, rdv_y), axis=0)

    count_train = {}
    count_train['true'] = 0
    count_train['false'] = 0
    count_train['unverified'] = 0
    for y in y_train:
        count_train[y] += 1

    count_test = {}
    count_test['true'] = 0
    count_test['false'] = 0
    count_test['unverified'] = 0
    for y in y_test:
        count_test[y] += 1

    print('count train', count_train)
    print('count test', count_test)

    print('x_train.shape', x_train.shape)
    print('x_test.shape', x_test.shape)
    print('y_train.shape', y_train.shape)
    print('y_test.shape', y_test.shape)

    tfidf = TfidfVectorizer()
    x_train = tfidf.fit_transform(x_train)
    x_test = tfidf.transform(x_test)

    rf = RandomForestClassifier(n_estimators=20)

    rf.fit(x_train, y_train)
    pred = rf.predict(x_test)
    print('len(pred)', len(pred))
    print('acc', accuracy_score(y_test, pred))
    print('y_test', y_test)
    print('pred', pred)


if __name__ == "__main__":
    baseline_model_tfidf()
