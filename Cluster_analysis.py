from sklearn.mixture import GaussianMixture
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from gensim import models, corpora
from matplotlib import pyplot as plt, pyplot
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

### read the dataset witb cluster labels
Total=pd.read_csv("/Users/qwertyui/我爱学习/540/project/DataFinal/Cluster_10.csv")
Total['Date_received'] = pd.to_datetime(Total['Date_received'], infer_datetime_format=True)
Total['month']= pd.DatetimeIndex(Total['Date_received']).month
Total['month_year']=Total['Date_received'].dt.to_period('M')

#########################################################################################################################################################
def random_sample(k,path):
    path = path + '/' + str(k) + '.csv'
    part = Total[Total.Cluster == k]
    p = part.sample(n=100)
    p.to_csv(path, index=False)

def histgram():
    size = []
    for i in range(0, 20):
        s = Total[Total.Cluster == i]
        print("Cluster ", i, " ", len(s))
        size.append(len(s))
    plt.bar(range(20), size)
    plt.show()

def get_tfidf_cluster():
    vectorizer = TfidfVectorizer(input='content', max_features=500, preprocessor=cleaning,
                                 ngram_range=(1, 2), stop_words="english", lowercase=True)
    vec = vectorizer.fit_transform(Total['Consumer_complaint_narrative'])
    dense = vec.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=vectorizer.get_feature_names())
    df['Cluster'] = Total['Cluster']
    return df

def cleaning(s):
    s = s.lower()
    #s=re.sub("x1*", "", s)              ## remove XX/XX/XXXX
    s=re.sub("x{2}(.*)", "", s)
    s=re.sub('[^A-Za-z]+', ' ', s)      ## remove special characters,punctuation
    return s

def wc(k,path,df):
    path = path + '/' + str(k) + '.png'
    s=df.copy()
    s = s[s["Cluster"] == k]
    s=s.drop(columns=["Cluster"])
    Cloud = WordCloud(background_color="white", max_words=80).generate_from_frequencies(s.T.sum(axis=1))
    plt.figure()
    plt.imshow(Cloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(path)

## Time series
def timeSeries(k,path):
    path = path + '/' + str(k) + '.png'
    ts=Total.groupby(['month_year']).size()
    s = Total[Total.Cluster == k]
    p = len(s) / len(Total)
    ts1=s.groupby(['month_year']).size()
    df1=pd.DataFrame(ts)
    df2=pd.DataFrame(ts1)
    df=pd.concat([df1,df2],axis=1)
    df.columns = ['Average', 'cluster ']
    df['Average'] = df['Average'] * p
    df.plot()
    title="Monthly complaint of Cluster {} .".format(k)
    plt.title(title)
    plt.savefig(path)
    ## Time series for Urgent
    ## Time series for Non-Urgent

###locations
def location(k,path):
    path = path + '/' + str(k) + '.png'
    stateList = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "GU", "HI", "IA", "ID", "IL", "IN",
                 "KS", "KY", "LA", "MA", "MD", "ME", "MH", "MI"
        , "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "PW",
                 "RI", "SC", "SD", "TN", "TX", "UT", "VA",
                 "VI", "VT", "WA", "WI", "WV", "WY"]
    T = Total[Total['State'].isin(stateList)]
    s = T[T.Cluster == k].copy()
    p = len(s) / len(T)
    geo_case = s.groupby(['State']).size()
    Total_case = T.groupby(['State']).size()
    df1 = pd.DataFrame(Total_case)
    df2 = pd.DataFrame(geo_case)
    df = pd.concat([df1, df2], axis=1)
    df.columns = ['Average', 'cluster 16']
    df['Average'] = df['Average'] * p
    df=df.round(2)
    ax = df.plot.barh(color={"red", "green"},figsize=(15,20))
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')
    title = "complaint by State of Cluster {} .".format(k)
    plt.title(title)
    plt.savefig(path)

def States():
    stateList = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "GU", "HI", "IA", "ID", "IL", "IN",
                 "KS", "KY", "LA", "MA", "MD", "ME", "MH", "MI"
        , "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "PW",
                 "RI", "SC", "SD", "TN", "TX", "UT", "VA",
                 "VI", "VT", "WA", "WI", "WV", "WY"]
    geo_case=Total[Total['State'].isin(stateList)]
    geo_case=geo_case.groupby(['State']).size()
    title = "complaint by State ."
    geo_case.plot(kind='bar', title=title)
    plt.show()

################################################################################################################################################################################################
#histgram()
#States()
#df=get_tfidf_cluster()
for k in range(0,10):
    print("processing cluster #",k)
    #random_sample(k, "/Users/qwertyui/我爱学习/540/project/DataFinal/samples")
    path="/Users/qwertyui/我爱学习/540/project/DataFinal/"
    #wc(k, path, df)
    #timeSeries(k, path)
    location(k, path)
    print("wordcloud #",k," is saved")
