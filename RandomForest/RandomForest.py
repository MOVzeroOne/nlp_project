#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import string
from sklearn import tree
#import seaborn as sns # Statistical data visualization
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# In[2]:


# If the rating is 1 or 2, classify as  the negative, labeled -1.
# If the rating is 4 or 5, then classify as the positive, labeled+1.
#All reviewsText with the rating 3 are neutral and will be ignored


# In[2]:


input_data = pd.read_csv("Grocery_and_Gourmet_Food.csv")
input_data['overall'] = input_data['overall'].astype(object)
input_data['reviewText'] = input_data['reviewText'].astype(object)

dt = {"reviewText": input_data["reviewText"], "overall": input_data["overall"]  }
dt = pd.DataFrame(data = dt)
dt = dt.dropna()

dt = dt[dt["overall"] != '3'] 
dt["label"] = dt["overall"].apply(lambda rating : "Positive" if str(rating) > '3' else "Negative")


# In[4]:


#Splitting the Dataset into training and testing


# In[5]:


#


# In[11]:


#sns.countplot(dt['label'], label = "Count")


# In[12]:


#sns.countplot(x = 'overall', data = dt)


# In[3]:



vectorizer = CountVectorizer()
gg = vectorizer.fit_transform(dt["reviewText"])


# In[ ]:


reviews = pd.DataFrame(gg.toarray())


# In[ ]:


dt = pd.concat([dt, reviews], axis=1)


# In[ ]:



X = pd.DataFrame(dt, columns = ["reviewText"])
y = pd.DataFrame(dt, columns = ["label"])

train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=50)




randomforest_classifier = RandomForestClassifier()
randomforest_classifier.fit(train_X, train_y)

y_predict_train = randomforest_classifier.predict(train_X)
cm = confusion_matrix(train_y, y_predict_train)
print(classification_report(train_y, y_predict_train))

randomforest_classifier.fit(test_X, test_y)

y_predict_test = randomforest_classifier.predict(test_X)
cm1 = confusion_matrix(test_y, y_predict_test)
print(classification_report(test_y, y_predict_test))