import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text

TotalPath = "/Users/qwertyui/我爱学习/540/project/DataFinal/Cleaned_random200 (1).csv"
savePath = "/Users/qwertyui/我爱学习/540/project/DataFinal/tf-idf_random200.csv"
Total = pd.read_csv(TotalPath)
Total.columns = [c.replace(' ', '_') for c in Total.columns]
documents = Total['Consumer_complaint_narrative'].values.tolist()

def cleaning(s):
    s=re.sub("X1*", "", s)              ## remove XX/XX/XXXX
    s=re.sub('[^A-Za-z]+', ' ', s)      ## remove special characters
    # tokens=[text for text in s.split()]
    # tokens=[WordNetLemmatizer().lemmatize(token) for token in tokens]   #lemmatize
    # tokens=[nltk.PorterStemmer().stem(token) for token in tokens]   # Stemming
    # s=' '.join(tokens)
    return s
# my_additional_stop_words=['ha', 'le', 'wa','abov', 'afterward', 'alon', 'alreadi', 'alway', 'ani', 'anoth', 'anyon', 'anyth', 'anywher', 'becam', 'becaus', 'becom',
#                           'befor', 'besid', 'cri', 'describ', 'dure', 'els', 'elsewher', 'empti', 'everi', 'everyon', 'everyth', 'everywher', 'fifti','formerli',
#                           'forti', 'henc', 'hereaft', 'herebi', 'hi', 'howev', 'hundr', 'inde', 'latterli', 'mani', 'meanwhil', 'moreov', 'mostli', 'nobodi', 'noon',
#                           'noth', 'nowher', 'onc', 'onli', 'otherwis', 'ourselv', 'perhap', 'pleas', 'seriou', 'sever', 'sinc','sincer', 'sixti', 'someon', 'someth',
#                           'sometim', 'somewher', 'themselv', 'thenc', 'thereaft', 'therebi', 'therefor', 'thi', 'thu', 'togeth', 'twelv','twenti', 'veri', 'whatev',
#                           'whenc', 'whenev', 'wherea', 'whereaft', 'wherebi', 'wherev', 'whi', 'yourselv']
#my_stop_words = text.ENGLISH_STOP_WORDS.union(my_additional_stop_words)
vectorizer = TfidfVectorizer(input='content',max_df=1.0, min_df=0.2, max_features=200,preprocessor=cleaning,
                             stop_words="english",lowercase=True) # stop words, punctuation are removed
tfidf = vectorizer.fit_transform(documents)
names=vectorizer.get_feature_names()