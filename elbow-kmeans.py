from sklearn.mixture import GaussianMixture
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from gensim import models
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction import text
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np
import gensim
from gensim import matutils

TotalPath = "/Users/qwertyui/我爱学习/540/project/DataFinal/part.csv"
tfidf_save_path="tf-idf.csv"
Total = pd.read_csv(TotalPath)
Total.columns = [c.replace(' ', '_') for c in Total.columns]
print(1)
def cleaning(s):
    s=re.sub("X1*", "", s)              ## remove XX/XX/XXXX
    s=re.sub('[^A-Za-z]+', ' ', s)      ## remove special characters,punctuation
    s=s.lower()
    tokens=[text for text in s.split()]
    # stopwords = nltk.corpus.stopwords.words('english')  # remove stop words
    # tokens = [t for t in tokens if not t in stopwords]   # remove stop words
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
vectorizer = TfidfVectorizer(input='content',max_df=0.8, min_df=0.2, max_features=200,preprocessor=cleaning,
                             stop_words="english",lowercase=True) # stop words
tfidf = vectorizer.fit_transform(Total['Consumer_complaint_narrative'])
print(2)
def save_tfidf():
    df=pd.DataFrame(tfidf.todense(),columns=vectorizer.get_feature_names())
    df.to_csv(tfidf_save_path,index=False)

def elbow_method():
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(1, 12))
    visualizer.fit(tfidf)
    visualizer.show()

def gaussian_method():
    k_arr = np.arange(0,100) + 1
    dense_tfidf=tfidf.todense()
    models = [
        GaussianMixture(n_components=k).fit(dense_tfidf)
        for k in k_arr
    ]
    AIC = [m.aic(dense_tfidf) for m in models]
    BIC = [m.bic(dense_tfidf) for m in models]
    # Plot these metrics
    plt.plot(k_arr, AIC, label='AIC')
    #plt.plot(k_arr, BIC, label='BIC')
    plt.xlabel('Number of Components ($k$)')
    plt.show()

"""最后会出现一个图片 把图片存下来就行了  感恩"""
def cluster_analysis1():
    k_optimum = 5
    km = KMeans(n_clusters=k_optimum, max_iter=100)
    km.fit(tfidf)
    clusters = km.labels_.tolist()
    Total['label'] = Total['Unnamed:_18'].map({'urgent': 1, 'non-urgent': 0})
    Total.drop(columns=['Unnamed:_18'])
    Total['Cluster'] = pd.Series(clusters)
    Total.to_csv("/Users/qwertyui/我爱学习/540/project/DataFinal/Cluster.csv")
    # lda = models.ldamodel.LdaModel(corpus=matutils.Sparse2Corpus(tfidf), id2word=vectorizer.vocabulary_, num_topics=100)
    ##WordCloud

    # for k in range(0, 5):
    #     s = Total[Total.Cluster == k]
    #     text = s['Consumer_complaint_narrative'].str.cat(sep=' ')
    #     text = text.lower()
    #     text = ' '.join([word for word in text.split()])
    #     wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
    #     print('Cluster: {}'.format(k))
    #     print('Titles')
    #     plt.figure()
    #     plt.imshow(wordcloud, interpolation="bilinear")
    #     plt.axis("off")
    #     plt.show()

gaussian_method()
#cluster_analysis1()