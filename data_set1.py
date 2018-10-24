import os
import pandas as pd
import json
import sklearn
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

DIR = 'data\\data\\rumoureval-2019-training-data\\reddit-training-data'

with open('data\\data\\rumoureval-2019-training-data\\train-key.json') as f:
    labels = json.load(f)
    labels = dict(labels['subtaskaenglish'])

prvi = True
za_izbacit = len('"data": {')
f2 = open('temp.json', 'w')
for filename in os.listdir(DIR):
    for filename2 in os.listdir(DIR + '\\' + filename):
        if filename2 == 'replies':
            for filename3 in os.listdir(DIR + '\\' + filename + '\\' + filename2):
                f = open(DIR + '\\' + filename + '\\' + filename2 + '\\' + filename3, 'r')
                for line in f.readlines():
                    pocetak = line.find('"data')
                    json = line[:pocetak] + line[pocetak+za_izbacit:-2] + ', "' + 'label' + '": "' + labels[filename3[:-5]] + '"}'
                    f2.write(json)
                    f2.write("\n")
                

df = pd.read_json('temp.json', lines=True)
print(df.shape)

'''
for xx in df[['body', 'label']].head(50).as_matrix():
    print(xx)
    print('____________________')
'''

x_text = df['body'].as_matrix()
y = df['label'].as_matrix()

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(x_text)

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20)
rf.fit(x, y)

print(rf.score(x, y))