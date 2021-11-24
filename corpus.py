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

"""This program takes in a File of uncleaned customer compliant csv,
 and returns a csv file with indexed, cleaned, tokenized compliant corpus and corresponding label
 (1=Urgent 0=Non_Urgent)"""
"""can also produce tf-idf by un-comment two lines """

TotalPath = "/Users/qwertyui/我爱学习/540/project/DataFinal/Cleaned_random200 (1).csv"
savePath = "/Users/qwertyui/我爱学习/540/project/DataFinal/corpus.csv"
#savePath="/Users/qwertyui/我爱学习/540/project/DataFinal/tf-idf.csv"
Total = pd.read_csv(TotalPath)
Total.columns = [c.replace(' ', '_') for c in Total.columns]
documents = Total['Consumer_complaint_narrative'].values.tolist()

###cleaning based on string
documents = [re.sub('[^A-Za-z0-9]+', ' ', doc) for doc in documents]    ###Remove Punctuations
# ocuments=[re.sub("X.*X", "", doc) for doc in documents]
documents = [doc.lower() for doc in documents]                          ###Convert to lower case
documents = [re.sub("\s+", " ", doc) for doc in documents]              ###Removing extra spaces

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

dictionary = corpora.Dictionary(texts)
mycorpus = [dictionary.doc2bow(doc, allow_update=True) for doc in texts]

tfidf = models.TfidfModel(mycorpus)
tfidf_vectors = [tfidf[word] for word in mycorpus]


output=Total[['Complaint_ID']].copy()
output['corpus']=pd.Series( mycorpus )
#output['tf-idf']=pd.Series( tfidf_vectors )
output['label']=Total['Unnamed:_18'].map({'urgent': 1, 'non-urgent': 0})

output.to_csv(savePath,index=False)