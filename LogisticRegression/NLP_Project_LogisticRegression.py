#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


import nltk
from nltk import word_tokenize 


# In[3]:


import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score


# In[4]:


df = pd.read_csv("Grocery_and_Gourmet_Food.csv")


# In[5]:


#df.info()


# In[19]:


ts = df.sample(10000)
ts


# In[28]:


ts = ts[ts["overall"] != '3'] 
ts["label"] = ts["overall"].apply(lambda rating : 1 if str(rating) > '3' else 0)
y = ts["label"]


# In[29]:


#text = " ".join(reviewText for reviewText in ts.reviewText.astype(str))


# In[30]:


vec = CountVectorizer()
X = vec.fit_transform(ts['reviewText'])
ts = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())


# In[31]:


ts


# In[34]:





ts_train, ts_test, y_train, y_test= train_test_split(ts,y)



# In[17]:



#wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
#plt.figure( figsize=(10,5))
#plt.imshow(wordcloud)
#plt.axis("off")
#plt.show()


# In[35]:


lr_model_all_new = LogisticRegression(max_iter = 200)
lr_model_all_new.fit(ts_train, y_train)

# Predicting the results
test_pred_lr_all = lr_model_all_new.predict(ts_test)

print(classification_report(y_test, test_pred_lr_all))




# In[ ]:




