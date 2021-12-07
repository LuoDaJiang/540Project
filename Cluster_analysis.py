import datetime
import os
import time

from wordcloud import WordCloud, STOPWORDS
import pandas as pd
import matplotlib.dates as mdates
from matplotlib import pyplot as plt, pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
import re


### read the dataset witb cluster labels
Total=pd.read_csv("/Users/qwertyui/我爱学习/540/project/DataFinal/Cluster_10.csv")
Total.columns = [c.replace(' ', '_') for c in Total.columns]
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
    for i in range(0, 10):
        s = Total[Total.Cluster == i]
        print("Cluster ", i, " ", len(s))
        size.append(len(s))
    plt.bar(range(10), size)
    plt.savefig("/Users/qwertyui/我爱学习/540/project/DataFinal/hist.png")
    plt.close()
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

def wordcloud(k,path,df):
    path = path + '/' + str(k) + '.png'
    s=df.copy()
    s = s[s["Cluster"] == k]
    s=s.drop(columns=["Cluster"])
    Cloud = WordCloud(background_color="white", max_words=80).generate_from_frequencies(s.T.sum(axis=1))
    plt.figure()
    plt.imshow(Cloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(path)
    print("keywords of #",k," ",Cloud.words_.keys())
    plt.close()
## Time series
def timeSeries1(k,path):
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
    plt.close()
    ## Time series for Urgent
    ## Time series for Non-Urgent

def timeSeries(k,path):
    path = path + '/' + str(k) + '.png'
    ts=Total.groupby(['month_year']).size()
    s = Total[Total.Cluster == k]
    p = len(s) / len(Total)
    ts1=s.groupby(['month_year']).size()
    df1=pd.DataFrame(ts)
    df2=pd.DataFrame(ts1)
    df=pd.concat([df1,df2],axis=1)
    df.columns = ['Average', "cluster {}".format(k)]
    df['Average'] = df['Average'] * p
    n = (  df["cluster {}".format(k)] -df['Average'])/ df['Average']
    n=n.round(2)
    #ax=n.plot(figsize=(16,9))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax = n.plot.barh(figsize=(15, 20))
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10),
                    textcoords='offset points')
    title="Monthly complaint of Cluster {} .".format(k)
    plt.title(title)
    plt.savefig(path)
    plt.close()
    ## Time series for Urgent
    ## Time series for Non-Urgent
###locations
def location1(k,path):
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
    c="cluster {}".format(k)
    df.columns = ['Average', "cluster {}".format(k)]
    df['Average'] = df['Average'] * p
    df=df.round(2)
    ax = df.plot.barh(color={"red", "green"},figsize=(15,20))
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')
    title = "complaint by State of Cluster {} .".format(k)
    plt.title(title)
    plt.savefig(path)
    plt.close()
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
    c="cluster {}".format(k)
    df.columns = ['Average', "cluster {}".format(k)]
    df['Average'] = df['Average'] * p
    n = (df["cluster {}".format(k)] -df['Average'] )/ df['Average']
    n=n.round(2)
    ax = n.plot.barh(figsize=(15,20))
    for p in ax.patches:
        ax.annotate(str(p.get_width()), (p.get_x() + p.get_width(), p.get_y()), xytext=(5, 10), textcoords='offset points')
    title = "complaint by State of Cluster {} .".format(k)
    plt.title(title)
    plt.savefig(path)
    plt.close()

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
    plt.close()

def urgent_rate(path):
    print("++++++++++++++++++++++++")
    dir_list = os.listdir(path)
    filepaths=[os.path.join(path,f) for f in dir_list]
    i=0
    for f in filepaths:
        df=pd.read_csv(f,encoding='latin-1')
        #df['label'] = df['Unnamed: 21'].map({'U': 1, 'N': 0})
        print("urgency rate of cluster#",i," is ",len(df[df['Unnamed: 21']=='U'])/200)
        i=i+1
    print("++++++++++++++++++++++++")
################################################################################################################################################################################################


histgram()
States()
print(datetime.datetime.now())
df=get_tfidf_cluster()
print(datetime.datetime.now())
WcParh="/Users/qwertyui/我爱学习/540/project/DataFinal/"
TsPath="/Users/qwertyui/我爱学习/540/project/DataFinal/"
LocationPath="/Users/qwertyui/我爱学习/540/project/DataFinal/"
for k in range(0,10):
    print("processing cluster #",k)
    random_sample(k, "/Users/qwertyui/我爱学习/540/project/DataFinal/samples")
    wordcloud(k, WcParh, df)
    timeSeries(k, TsPath)
    location(k, LocationPath)
    print("Cluster #",k," is processed")
    pass
path="/Users/qwertyui/我爱学习/540/project/DataFinal/samples-labeled"
urgent_rate(path)

