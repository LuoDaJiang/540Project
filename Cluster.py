from sklearn.mixture import GaussianMixture
from wordcloud import WordCloud, STOPWORDS
import pandas as pd
from gensim import models
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


#preprocessing and generate tf-id
########################################################################################################################
#private variavles
TotalPath = "/Users/qwertyui/我爱学习/540/project/DataFinal/complaints_Cleaned.csv"
Total = pd.read_csv(TotalPath)
Total.columns = [c.replace(' ', '_') for c in Total.columns]
Total['Date_received'] = pd.to_datetime(Total['Date_received'], infer_datetime_format=True)
Total['month']= pd.DatetimeIndex(Total['Date_received']).month
Total['month_year']=Total['Date_received'].dt.to_period('M')
print(1)
# ts=Total.groupby(['month_year']).size()
# ts.plot(label="all")
# pyplot.title("Monthly complaint")
# pyplot.show()

def cleaning(s):
    s = s.lower()
    #s=re.sub("x1*", "", s)              ## remove XX/XX/XXXX
    s=re.sub("x{2}(.*)", "", s)
    s=re.sub('[^A-Za-z]+', ' ', s)      ## remove special characters,punctuation
    return s

vectorizer = TfidfVectorizer(input='content', max_df=0.8, min_df=0.2, max_features=500, preprocessor=cleaning,
                                 stop_words="english", lowercase=True)  # stop words
tfidf = vectorizer.fit_transform(Total['Consumer_complaint_narrative'])
#global variables:Total vectorizer tfidf
print(2)
########################################################################################################################

def save_tfidf(tfidf,path):
    df=pd.DataFrame(tfidf.todense(),columns=vectorizer.get_feature_names())
    df.to_csv(path,index=False)

# optimum cluster number and clustering
def elbow_method(k1,k2):
    model = KMeans()
    visualizer = KElbowVisualizer(model, k=(k1,k2))
    visualizer.fit(tfidf)
    visualizer.show()

def gaussian_method(path,k):
    k_arr = np.arange(0,k) + 1
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
    #plt.show()
    plt.savefig(path)
    return min(AIC)
    """最后会出现一个图片 把图片存下来就行了  感恩"""


def gaussian_single(k):
    dense_tfidf = tfidf.todense()
    models = GaussianMixture(n_components=k).fit(dense_tfidf)
    AIC = models.aic(dense_tfidf)
    return AIC

def save_cluster(k_optimum,path):
    km = KMeans(n_clusters=k_optimum, max_iter=100)
    km.fit(tfidf)
    clusters = km.labels_.tolist()
    #Total['label'] = Total['Unnamed:_18'].map({'urgent': 1, 'non-urgent': 0})
    #Total.drop(columns=['Unnamed:_18'])
    Total['Cluster'] = pd.Series(clusters)
    Total.to_csv(path,index=False)
    return Total
    # lda = models.ldamodel.LdaModel(corpus=matutils.Sparse2Corpus(tfidf), id2word=vectorizer.vocabulary_, num_topics=100)



#########################################################################################################################################################
#functions below are only callable after call save_cluster()
def wordCloud(k_optimum):
    ##WordCloud
    for k in range(0, k_optimum):
        s = Total[Total.Cluster == k]
        text = s['Consumer_complaint_narrative'].str.cat(sep=' ')
        text = text.lower()
        text = re.sub("x1*", "", text)  ## remove XX/XX/XXXX
        text = re.sub('[^A-Za-z]+', ' ', text)  ## remove special characters,punctuation
        text = ' '.join([word for word in text.split()])
        stopwords=set(STOPWORDS)
        wordcloud = WordCloud(max_font_size=50, max_words=100,stopwords=stopwords, background_color="white").generate(text)
        print('Cluster: {}'.format(k))
        print('Titles')
        plt.figure()
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()
    ## WordCloud for Urgent
    ## WordCloud for Non-Urgent
def single_wordCloud(k):
    s = Total[Total.Cluster == k]
    text = s['Consumer_complaint_narrative'].str.cat(sep=' ')
    text = text.lower()
    text = re.sub("x1*", "", text)  ## remove XX/XX/XXXX
    text = re.sub('[^A-Za-z]+', ' ', text)  ## remove special characters,punctuation
    text = ' '.join([word for word in text.split()])
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(max_font_size=50, max_words=100, stopwords=stopwords, background_color="white").generate(text)
    print('Cluster: {}'.format(k))
    print('Titles')
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.show()

def wc(k):
    s = Total[Total.Cluster == k]
    vec = vectorizer.fit_transform(s['Consumer_complaint_narrative'])
    dense= vec.todense()
    lst1 = dense.tolist()
    df = pd.DataFrame(lst1, columns=vectorizer.get_feature_names())
    Cloud = WordCloud(background_color="white", max_words=50).generate_from_frequencies(df.T.sum(axis=1))

## Time series
def timeSeries(k_optimum):
    ts_list=[]
    for k in range(0, k_optimum):
        s = Total[Total.Cluster == k]
        ts = s.groupby(['Date_received']).size()
        ts_list.append(ts)
    ts_list.plot(label=k_optimum,subplots=True, legend=False)
    plt.show()

def single_timeSeries(k):
    ts=Total.groupby(['month_year']).size()
    s = Total[Total.Cluster == k]
    p = len(s) / len(Total)
    ts1=s.groupby(['month_year']).size()
    df1=pd.DataFrame(ts)
    df2=pd.DataFrame(ts1)
    df=pd.concat([df1,df2],axis=1)
    df.columns = ['Average', 'cluster 16']
    df['Average'] = df['Average'] * p
    df.plot()
    title="Monthly complaint of Cluster {} .".format(k)
    plt.title(title)
    plt.show()
    ## Time series for Urgent
    ## Time series for Non-Urgent

###locations
def location(k):
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
    ax = df.plot.barh(color={"red", "green"})
    title = "complaint by State of Cluster {} .".format(k)
    plt.title(title)
    plt.show()
def single_location(k):
    ## Location
    stateList = ["AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC", "DE", "FL", "GA", "GU", "HI", "IA", "ID", "IL", "IN",
                 "KS", "KY", "LA", "MA", "MD", "ME", "MH", "MI"
        , "MN", "MO", "MS", "MT", "NC", "ND", "NE", "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "PR", "PW",
                 "RI", "SC", "SD", "TN", "TX", "UT", "VA",
                 "VI", "VT", "WA", "WI", "WV", "WY"]
    T = Total[Total['State'].isin(stateList)]
    s = T[T.Cluster == k].copy()
    p=len(s)/len(T)
    geo_case=s.groupby(['State']).size()
    Total_case=T.groupby(['State']).size()
    avg_case=Total_case.values*p
    #geo_case.plot(kind='bar', title=title)
    plt.bar(geo_case.index.values,geo_case.values,label=k)
    plt.bar(Total_case.index.values,avg_case,label="average")
    title = "complaint by State of Cluster {} .".format(k)
    plt.title(title)
    plt.show()
    ## Location for Urgent
    ## Location for Non-Urgent

def histgram():
    size = []
    for i in range(0, 20):
        s = Total[Total.Cluster == i]
        print("Cluster ", i, " ", len(s))
        size.append(len(s))
    plt.bar(range(20), size)
    plt.show()

def random_sample():
    part = Total[Total.Cluster == 16]
    p = part.sample(n=100)
    p.to_csv("/Users/qwertyui/我爱学习/540/project/DataFinal/test_16.csv", index=False)
def cluster_analysis(k):
    part = Total[Total.Cluster == k]
    print("size ",len(part))
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


# Total=pd.read_csv("/Users/qwertyui/Downloads/Cluster.csv")
# Total['Date_received'] = pd.to_datetime(Total['Date_received'], infer_datetime_format=True)
# Total['month']= pd.DatetimeIndex(Total['Date_received']).month
# Total['month_year']=Total['Date_received'].dt.to_period('M')
#save_tfidf(tfidf)
#elbow_method()
#k_optimum=gaussian_method('/Users/qwertyui/我爱学习/540/project/DataFinal/AIC.png')
Total=save_cluster(10,"/Users/qwertyui/我爱学习/540/project/DataFinal/Cluster_10.csv")
wc(0)
#timeSeries(20)
#single_timeSeries(16)

#single_location(16)
#histgram()
#wordCloud(20)
#Total=save_cluster(k_optimum)
#wordCloud(k_optimum)

#States()
#cluster_analysis(16)
#location(0)