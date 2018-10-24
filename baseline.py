import pandas as pd
import json
import sklearn
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_json('temp.json', lines=True)
print(df.shape)

'''
for xx in df[['body', 'label']].head(50).as_matrix():
    print(xx)
    print('____________________')
'''

valid_rows = [pd.notnull(body) and body != '[deleted]' and body != '[removed]' for body in df['body']]
df = df[valid_rows]
print(df.shape)
x_text = df['body'].as_matrix()
y = df['label'].as_matrix()

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(x_text)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2)
rf = RandomForestClassifier(n_estimators=20)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
print(accuracy_score(y_test, pred))
print(pred)