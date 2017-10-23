import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Convolution1D, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential, load_model
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.metrics import f1_score
from textblob import TextBlob

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# function to clean data
stops = set(stopwords.words("english"))

def cleanData(text):
    txt = str(text)
    txt = re.sub(r'[^A-Za-z0-9\s]', r'', txt)
    txt = re.sub(r'\n', r' ', txt)
    txt = " ".join([w for w in txt.split() if w not in stops])
    temp = re.sub(r'[^A-Z]', r' ', txt)
    txt = " ".join([w.lower() for w in txt.split() if w not in temp])
    txt = temp + " " + txt
    txt = ' '.join(word for word in txt.split() if len(word)>1)
    st = PorterStemmer()
    txt = " ".join([st.stem(w) for w in txt.split()])
    return txt

def sentenceSentiment(text):
    txt=text.split('.')
    pol=0
    subj=0
    for i in range(len(txt)):
        temp=TextBlob(txt[i])
        pol=pol+temp.sentiment[0]
        subj=subj+temp.sentiment[1]
    return [pol,subj]

## join data
test['Is_Response'] = np.nan
alldata = pd.concat([train, test]).reset_index(drop=True)
y_alldata = [1 if x == 'happy' else 0 for x in alldata['Is_Response']]
polarity=[]
subjectivity=[]
for i in range(len(alldata)):
    temp= sentenceSentiment(alldata.ix[i,'Description'])
    polarity.append(temp[0])
    subjectivity.append(temp[1])

alldata['Description'] = alldata['Description'].map(lambda x: cleanData(x))
alldata['polarity']=polarity
alldata['subjectivity']=subjectivity

cols = ['Browser_Used','Device_Used']
from sklearn.preprocessing import LabelEncoder
for x in cols:
    lbl = LabelEncoder()
    alldata[x] = lbl.fit_transform(alldata[x])

MAX_SEQUENCE_LENGTH = 1000
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100

print('Indexing word vectors.')

embeddings_index = {}
f = open('glove.6B.100d.txt', encoding="utf8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

texts = np.array(alldata['Description'])  # list of text samples
labels = [0.0,1.0]  # list of label ids
print('Found %s texts.' % len(texts))

# vectorizing the text samples into a 2D integer tensor
tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

df=data
df = pd.DataFrame(df)
df.columns = ['col'+ str(x) for x in df.columns]
df_all=alldata[cols]
df_all=pd.get_dummies(df_all, columns=['Device_Used', 'Browser_Used'])
df_all.drop(['Browser_Used_9','Browser_Used_10'], axis=1, inplace=True) # less than 1% 
df_all['polarity']=alldata['polarity']
df_all['subjectivity']=alldata['subjectivity']
df = pd.concat([df_all*100, df], axis = 1)
labels = to_categorical(np.asarray(labels))
print('Shape of data tensor:', df.shape)
print('Shape of label tensor:', labels.shape)

X_train = np.array(df[:int(0.95*len(train))])
X_dev= np.array(df[int(0.95*len(train)):int(len(train))])
X_test= np.array(df[int(len(train)):])
target = y_alldata
Y_train = target[:int(0.95*len(train))]
Y_dev = target[int((0.95*len(train))):int(len(train))]
Y_test = target[int(len(train)):]

# prepare embedding matrix
num_words = len(word_index)
embedding_matrix = np.zeros((num_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if i >= MAX_NB_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print("Preparing Model!")

# best model
model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=df.shape[1]))
model.add(Conv1D(nb_filter=EMBEDDING_DIM, filter_length=5, border_mode='same', activation='relu'))
model.add(MaxPooling1D(2))
model.add(Flatten())
model.add(Dropout(0.9))
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])

flag=True
BATCH=128
while flag:
    if BATCH==2048 break
    model.fit(X_train, Y_train, validation_data=(X_dev, Y_dev), nb_epoch=1, batch_size=BATCH) # epochs=3, batch=128+512+2048
    scores = model.evaluate(X_dev, Y_dev, verbose=0)
    print("Accuracy: %.5f%%" % (scores[1]*100))
    BATCH=BATCH*4
    
model.save('model.h5')

def to_labels(x):
    if x > 0.5:  # cutoff
        return "happy"
    return "not_happy"

submission = model.predict(X_test)
sub=[]
for i in range(len(submission)):
    sub.append(submission[i][0])

submission_data = pd.DataFrame({'User_ID':test.User_ID, 'Is_Response':sub})
submission_data['Is_Response'] = submission_data['Is_Response'].map(lambda x: to_labels(x))
submission_data = submission_data[['User_ID','Is_Response']]
submission_data.to_csv("submission.csv", index=False) # 0.88 score
