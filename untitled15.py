# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 14:38:44 2024

@author: Pratham
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer,CountVectorizer

email_data=pd.read_csv("C:/naive bayes classification/sms_raw_NB.csv",encoding='ISO-8859-1')

import re

def cleaning_text(i):
    w=[]
    i=re.sub('[^A-Za-z""]+',' ',i).lower()
    for word in i.split(' '):
        if len(word) > 3:
            w.append(word)
        return (' '.join(w))
    
cleaning_text('Hope you are having good week.just checking')
cleaning_text('hope i can understand your feelings 123121.123.hi how are you')
cleaning_text('hi how are you ,i am sad')

email_data.text=email_data.text.apply(cleaning_text)
email_data=email_data.loc[email_data.text !="",:]

from sklearn.model_selection import train_test_split
email_train,email_test=train_test_split(email_data,test_size=0.2)
#creating matrix of token counts for entire text document

def split_into_words(i):
    return [word for word in i.split(" ")]

email_bow=CountVectorizer(analyzer=split_into_words).fit(email_data.text)

all_emails_matrix=email_bow.transform(email_data.text)

train_emails_matrix=email_bow.transform(email_train.text)

test_emails_matrix=email_bow.transform(email_test.text)

tfidf_transformer=TfidfTransformer().fit(all_emails_matrix)

train_tfidf=tfidf_transformer.transform(train_emails_matrix)

test_tfidf=tfidf_transformer.transform(test_emails_matrix)

test_tfidf.shape

from sklearn.naive_bayes import MultinomialNB as MB

classifier_mb=MB()
classifier_mb.fit(train_tfidf,email_train.type)

test_pred_m=classifier_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.type)
accuracy_test_m

