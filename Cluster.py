from sklearn.mixture import GaussianMixture
import pandas as pd
from matplotlib import pyplot as plt, pyplot
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import numpy as np


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
    s=re.sub("x{2}(.*)", "", s)         ## remove XX/XX/XXXX
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

save_tfidf(tfidf)
elbow_method()
k_optimum=10
savepath="/Users/qwertyui/Downloads/Cluster.csv"
save_cluster(k_optimum,savepath)