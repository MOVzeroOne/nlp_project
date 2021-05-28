import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("Grocery_and_Gourmet_Food.csv")
ts = df.sample(10000)

ts = ts[ts["overall"] != '3']
ts["label"] = ts["overall"].apply(lambda rating : 1 if str(rating) > '3' else 0)
y = ts["label"]

vec = CountVectorizer()
X = vec.fit_transform(ts['reviewText'])
ts = pd.DataFrame(X.toarray(), columns = vec.get_feature_names())


ts_train, ts_test, y_train, y_test= train_test_split(ts,y)

kNearN_classifier = KNeighborsClassifier()
kNearN_classifier.fit(ts_train, y_train)

# Predicting the results
kNearN_classifier_prediction = kNearN_classifier.predict(ts_test)


print(classification_report(y_test, kNearN_classifier_prediction))