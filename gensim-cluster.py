if __name__ == '__main__':
    from multiprocessing import freeze_support
    import pandas as pd
    import gensim
    from gensim import corpora
    import re
    import nltk
    from nltk.stem import WordNetLemmatizer
    from gensim import models
    from spellchecker import SpellChecker
    import csv
    from gensim.models import LdaModel, LdaMulticore
    from distributed import Client
    from sklearn.cluster import KMeans
    from gensim.matutils import corpus2dense, corpus2csc

    TotalPath = "/Users/qwertyui/我爱学习/540/project/DataFinal/Cleaned_random200 (1).csv"
    savePath = "/Users/qwertyui/我爱学习/540/project/DataFinal/tf-idf_random200.csv"


    Total = pd.read_csv(TotalPath)
    Total.columns = [c.replace(' ', '_') for c in Total.columns]
    # na.to_csv(savePath,index=False)
    documents = Total['Consumer_complaint_narrative'].values.tolist()
    # documents=narrative[0:10]

    ###cleaning based on string
    documents = [re.sub('[^A-Za-z0-9]+', ' ', doc) for doc in documents]  ###Remove Punctuations
    documents=[re.sub("X1*", "", doc) for doc in documents]     ## remove XX/XX/XXXX
    documents = [doc.lower() for doc in documents]  ###Convert to lower case
    documents = [re.sub("\s+", " ", doc) for doc in documents]  ###Removing extra spaces

    ### Tokenize(split) the sentences into words
    texts = [[text for text in doc.split()] for doc in documents]

    # cleaning based on words
    stopwords = nltk.corpus.stopwords.words('english')  # remove stop words
    texts = [[t for t in text if not t in stopwords] for text in texts]  # remove stop words
    ps = nltk.PorterStemmer()  # Stemming
    texts = [[ps.stem(word) for word in text] for text in texts]  # Stemming
    le = WordNetLemmatizer()  # lemmatize
    texts = [[le.lemmatize(word) for word in text] for text in texts]  # lemmatize
    # spell = SpellChecker()                                              #spell checker
    # texts = [[spell.correction(word) for word in text] for text in texts]#spell checker

    ### get tf-idf
    dictionary = corpora.Dictionary(texts)
    mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]
                                                        # print(dictionary)
    print(dictionary.token2id)
    tfidf = models.TfidfModel(mycorpus)
    tfidf_vectors = [tfidf[word] for word in mycorpus]
    corpus_tfidf_dense = corpus2dense(mycorpus, 2 * len(dictionary), num_docs=len(mycorpus))
    corpus_tfidf_sparse = corpus2csc(mycorpus, 2 * len(dictionary), num_docs=len(mycorpus))

    ### cluster
    km = KMeans(n_clusters=5)
    km.fit(corpus_tfidf_dense.T)
    clusters = km.labels_.tolist()

    ### save as csv
    Total['Cluster']=pd.Series(clusters)
    output = Total[['Complaint_ID','Cluster']].copy()
    output['label'] = Total['Unnamed:_18'].map({'urgent': 1, 'non-urgent': 0})
    output.to_csv(savePath, index=False)

    # lda_model = LdaMulticore(corpus=mycorpus,
    #                          id2word=dictionary,
    #                          random_state=100,
    #                          num_topics=7,
    #                          passes=10,
    #                          chunksize=1000,
    #                          batch=False,
    #                          alpha='asymmetric',
    #                          decay=0.5,
    #                          offset=64,
    #                          eta=None,
    #                          eval_every=0,
    #                          iterations=100,
    #                          gamma_threshold=0.001,
    #                          per_word_topics=True)

