import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer

TotalPath = "/Users/qwertyui/我爱学习/540/project/DataFinal/Cleaned_random200 (1).csv"
savePath = "/Users/qwertyui/我爱学习/540/project/DataFinal/tf-idf_random200.csv"
Total = pd.read_csv(TotalPath)
Total.columns = [c.replace(' ', '_') for c in Total.columns]
documents = Total['Consumer_complaint_narrative'].values.tolist()

def cleaning(s):
    s=re.sub("X1*", "", s)              ## remove XX/XX/XXXX
    s=re.sub('[^A-Za-z]+', ' ', s)      ## remove special characters
    tokens=[text for text in s.split()]
    tokens=[nltk.PorterStemmer().stem(token) for token in tokens]   # Stemming
    tokens=[WordNetLemmatizer().lemmatize(token) for token in tokens]   #lemmatize
    s=[' '.join(tokens) ]
    return s
vectorizer = TfidfVectorizer(input='content',max_df=0.8, min_df=0.2, max_features=200,preprocessor=cleaning,
                             stop_words='english',lowercase=True) # stop words, punctuation are removed
tfidf = vectorizer.fit_transform(documents)
names=vectorizer.get_feature_names()