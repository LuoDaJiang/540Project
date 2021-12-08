import pandas as pd
import gensim
from gensim import corpora
from gensim import models
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import sklearn.metrics

TotalPath="1_labeled.csv"
trainPath = "complaints_Cleaned_random200.csv"

Total=pd.read_csv(TotalPath)
test_y = Total.iloc[:, -1]
test_y = test_y.map({'N': 0, 'U': 1})
test_y = test_y.to_numpy()
Total.columns = [c.replace(' ', '_') for c in Total.columns]
na=Total['Consumer_complaint_narrative']
# na.to_csv(savePath,index=False)
documents=na.values.tolist()

# documents=narrative[0:10]
train = pd.read_csv(trainPath)
y = train.iloc[:, -1]
train_y = y.map({'non-urgent': 0, 'urgent': 1})
train_y = train_y.to_numpy()
train.columns = [c.replace(' ', '_') for c in train.columns]
train_na = train['Consumer_complaint_narrative']
train_documents = train_na.values.tolist()

#data cleanning
def cleaning(s):
    s=re.sub("X1*", "", s)              ## remove XX/XX/XXXX
    s=re.sub('[^A-Za-z]+', ' ', s)      ## remove special characters,punctuation
    s=s.lower()
    tokens=[text for text in s.split()]
    # tokens=[WordNetLemmatizer().lemmatize(token) for token in tokens]   #lemmatize
    # tokens=[nltk.PorterStemmer().stem(token) for token in tokens]   # Stemming
    # s=' '.join(tokens)
    return s

def documentcleaning(documents):
    documents = [re.sub('[^A-Za-z0-9]+', ' ', doc) for doc in documents]
    documents = [re.sub("X1*", ' ', doc) for doc in documents]
    documents=[doc.lower() for doc in documents]
    documents=[re.sub("\s+"," ",doc) for doc in documents]
    return documents

#calculate the average lenth of train document
cleaned_train = documentcleaning(documents=train_na)
train_len = len(cleaned_train)
length = 0
for i in range(train_len):
    length += len(cleaned_train[i])
avg_len = length/train_len


def lengthmodel(X):
    y_predict = []
    for i in range (len(X)):
        if len(X[i])> avg_len:
            y_predict.append(1)
        else:
            y_predict.append(0)
    np.asarray(y_predict)
    return y_predict

length_predict = lengthmodel(X=documentcleaning(documents=na))

# for tf-idf model
nltk_stopwords = set(stopwords.words('english'))
stop_words=nltk_stopwords
tf_idf_train_x=pd.Series(train_documents)
vectorizer = TfidfVectorizer(input='content', max_df=0.8, min_df=0.2,max_features=500, preprocessor=cleaning,
                             stop_words=stop_words, lowercase=True)
vectors = vectorizer.fit(tf_idf_train_x)
vectors_train = vectors.transform(tf_idf_train_x)
tf_idf_train_X = pd.DataFrame(vectors_train.todense().tolist(), columns=vectorizer.get_feature_names())

tf_idf_test_x=pd.Series(documents)
test_vectors = vectors.transform(tf_idf_test_x)
tf_idf_test_X = pd.DataFrame(test_vectors.todense().tolist(), columns=vectorizer.get_feature_names())

vectorizer.get_feature_names_out()
# for word2vec
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "parser", "attribute_ruler"])
train_vectors = []
for idx, msg in enumerate(train_na): # iterate through each review
    train_vectors.append(nlp(msg).vector)
X=pd.DataFrame(train_vectors)
index=X.loc[X.isin([np.nan, np.inf, -np.inf]).any(1)].index
word2vec_train_X=X.drop(index)
word2vec_train_X = word2vec_train_X.values

nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "parser", "attribute_ruler"])
test_vectors = []
for idx, msg in enumerate(na): # iterate through each review
    test_vectors.append(nlp(msg).vector)
test_X=pd.DataFrame(test_vectors)
test_index=test_X.loc[X.isin([np.nan, np.inf, -np.inf]).any(1)].index
word2vec_test_X=test_X.drop(test_index)
word2vec_test_X = word2vec_test_X.values


# random label
random_predict_y = []
labels = [0,1]
for i in range(100):
    random_predict_y.append(random.choice(labels))
random_predict_y = np.asarray(random_predict_y)



def train_models(X,y,model):
    if model == 'decision_tree':
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(X, y)
    if model == 'naive_bayes':
        clf = BernoulliNB()
        clf = clf.fit(X, y)
    if model == 'linear_SGD_classifier':
        clf = sklearn.linear_model.SGDClassifier(loss='squared_error')
        clf = clf.fit(X, y)
    if model == 'random_forest':
        clf = RandomForestClassifier()
        clf = clf.fit(X, y)
    if model == 'logistic_regression':
        clf = LogisticRegression()
        clf = clf.fit(X, y)
    return clf

# cross validation
def cross_validation_score(splits,X,y):
    kfold = StratifiedKFold(n_splits=splits, random_state=None, shuffle=False)
    m = ['decision_tree', 'naive_bayes', 'linear_SGD_classifier', 'random_forest', 'logistic_regression']
    score_list = []
    for i in m:
        clf = train_models(X=X, y=y, model = i)
        score = cross_val_score(clf, X=X, y=y, cv=kfold)
        score_mean = score.mean()
        score_list.append(score_mean)
    return score_list

# metric table
def accuracy_evaluate_model(X_test, y_test,X_train, y_train):
    m = ['decision_tree', 'naive_bayes', 'linear_SGD_classifier', 'random_forest', 'logistic_regression']
    for i in m:
        clf = train_models(X_train, y_train, model=i)
        y_predict = clf.predict(X_test)
        print("Model:"+i)
        print(sklearn.metrics.classification_report(y_test, y_predict))

print("***metric - tf-idf***")
accuracy_evaluate_model(X_train=tf_idf_train_X, y_train = train_y,X_test=tf_idf_test_X, y_test=test_y)
print("***metric***-word2vec")
accuracy_evaluate_model(X_train=word2vec_train_X, y_train = train_y,X_test=word2vec_test_X, y_test=test_y)
print("***cross-validation score - tf-idf***")
cross_validation_score(splits=10,X=tf_idf_train_X,y=train_y)
print("***cross-validation score***-word2vec")
cross_validation_score(splits=10,X=word2vec_train_X,y=train_y)
print("***random model***")
print(sklearn.metrics.classification_report(test_y,random_predict_y))
print("***length model***")
print(sklearn.metrics.classification_report(test_y,length_predict))