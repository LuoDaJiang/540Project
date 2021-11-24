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

TotalPath="/Users/qwertyui/我爱学习/540/project/DataFinal/Cleaned_random200 (1).csv"
savePath="/Users/qwertyui/我爱学习/540/project/DataFinal/tf-idf_random200.csv"
narrativepath="/Users/qwertyui/我爱学习/540/project/DataFinal/narratives.csv"

Total=pd.read_csv(TotalPath)
Total.columns = [c.replace(' ', '_') for c in Total.columns]
na=Total['Consumer_complaint_narrative']
#na.to_csv(savePath,index=False)
documents=na.values.tolist()
documents=documents[:10]
# documents=narrative[0:10]

###cleaning based on string
documents = [re.sub('[^A-Za-z0-9]+', ' ', doc) for doc in documents]    ###Remove Punctuations
#ocuments=[re.sub("X.*X", "", doc) for doc in documents]
documents=[doc.lower() for doc in documents]                            ###Convert to lower case
documents=[re.sub("\s+"," ",doc) for doc in documents]                  ###Removing extra spaces

### Tokenize(split) the sentences into words
texts = [[text for text in doc.split()] for doc in documents]

#cleaning based on words
stopwords = nltk.corpus.stopwords.words('english')                  #remove stop words
texts=[[t for t in text if not t in stopwords]for text in texts]    #remove stop words
ps = nltk.PorterStemmer()                                           #Stemming
texts = [[ps.stem(word) for word in text] for text in texts]        #Stemming
le = WordNetLemmatizer()                                            #lemmatize
texts = [[le.lemmatize(word) for word in text] for text in texts]   #lemmatize
# spell = SpellChecker()                                              #spell checker
# texts = [[spell.correction(word) for word in text] for text in texts]#spell checker

dictionary = corpora.Dictionary(texts)
mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]
#print(dictionary)
#print(dictionary.token2id)

tfidf = models.TfidfModel(mycorpus)
vectors = [tfidf[word] for word in mycorpus]
with open(savePath, 'w') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)

    write.writerow(['tf-idf'])
    write.writerows(vectors)


lda_model = LdaMulticore(corpus=mycorpus,
                         id2word=dictionary,
                         random_state=100,
                         num_topics=7,
                         passes=10,
                         chunksize=1000,
                         batch=False,
                         alpha='asymmetric',
                         decay=0.5,
                         offset=64,
                         eta=None,
                         eval_every=0,
                         iterations=100,
                         gamma_threshold=0.001,
                         per_word_topics=True)
