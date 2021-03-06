import re
import string
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

def getdocs():
  r = requests.get('https://www.kompas.com/')
  soup = BeautifulSoup(r.content, 'html.parser')

  link = []
  for i in soup.find('div', {'class': 'most__wrap'}).find_all('a'):
      i['href'] = i['href'] + '?page=all'
      link.append(i['href'])

# Getting the paragraphs
  doc = []
  for i in link:
      r = requests.get(i)
      soup = BeautifulSoup(r.content, 'html.parser')

      sen = []
      for i in soup.find('div', {'class': 'read__content'}).find_all('p'):
          sen.append(i.text)
      doc.append(' '.join(sen))

  # Cleaning up paragraphs
  doc_clean = []
  for d in doc:
      doc_test = re.sub(r'[^\x00-\x7F]+', ' ', d)
      doc_test = re.sub(r'@\w+', '', doc_test)
      doc_test = doc_test.lower()
      doc_test = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', doc_test)
      doc_test = re.sub(r'[0-9]', '', doc_test)
      doc_test = re.sub(r'\s{2,}', ' ', doc_test)
      doc_clean.append(doc_test)

  return doc_clean

docs = getdocs()

# TF-IDF weighting
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(docs)

# Creating dataFrame
df = pd.DataFrame(X.T.toarray(), index=vectorizer.get_feature_names())
docs = getdocs()

def getArticles(q, df):
  print("Search term:", q)
  q = [q]
  q_vec = vectorizer.transform(q).toarray().reshape(df.shape[0])
  sim = {}
  for i in range(10):
    sim[i] = np.dot(df.loc[:, i].values, q_vec) / np.linalg.norm(df.loc[:, i]) * np.linalg.norm(q_vec)
  
  sim_sorted = sorted(sim.items(), key=lambda x: x[1], reverse=True)
  
  for k, v in sim_sorted:
    if v != 0.0:
      print("Similar article (cosine sim. value): ", v)
      print(docs[k])
      print()

terms = 'covid-19'

getArticles(terms, df)
print('-'*100)