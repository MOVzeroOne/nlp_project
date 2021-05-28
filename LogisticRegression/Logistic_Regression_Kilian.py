import matplotlib.pyplot as plt
import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import sklearn.model_selection
import warnings
warnings.filterwarnings("ignore")

# Clean reviews
def clean_reviews(review, remove_stopwords=True):
    # Convert words to lower case
    review = review.lower()
    # Replace contractions with their longer forms
    if True:
        review = review.split()
        new_text = []
        for word in review:
            if word in contractions:
                new_text.append(contractions[word])
            else:
                new_text.append(word)
        review = " ".join(new_text)
    # Format words and remove unwanted characters
    review = re.sub(r'https?:\/\/.*[\r\n]*', '', review, flags=re.MULTILINE)
    review = re.sub(r'\<a href', ' ', review)
    review = re.sub(r'&amp;', '', review)
    review = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/]', ' ', review)
    review = re.sub(r'<br />', ' ', review)
    review = re.sub(r'\'', ' ', review)
    # Remove stopwords
    if remove_stopwords:
        review = review.split()
        stops = set(stopwords.words("english"))
        review = [w for w in review if not w in stops]
        review = " ".join(review)
    # Tokenize the words
    review = nltk.WordPunctTokenizer().tokenize(review)
    return review

# Lemmatize reviews
def lemmatize_words(review):
    lemm = nltk.stem.WordNetLemmatizer()
    df['lemmatized_text'] = list(map(lambda word: list(map(lemm.lemmatize, word)), df.Cleaned_reviews))

if __name__ == '__main__':
    # A list of contractions from: http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
    contractions = {
        "ain't": "am not",
        "aren't": "are not",
        "can't": "cannot",
        "can't've": "cannot have",
        "'cause": "because",
        "could've": "could have",
        "couldn't": "could not",
        "couldn't've": "could not have",
        "didn't": "did not",
        "doesn't": "does not",
        "don't": "do not",
        "hadn't": "had not",
        "hadn't've": "had not have",
        "hasn't": "has not",
        "haven't": "have not",
        "he'd": "he would",
        "he'd've": "he would have",
        "he'll": "he will",
        "he's": "he is",
        "how'd": "how did",
        "how'll": "how will",
        "how's": "how is",
        "i'd": "i would",
        "i'll": "i will",
        "i'm": "i am",
        "i've": "i have",
        "isn't": "is not",
        "it'd": "it would",
        "it'll": "it will",
        "it's": "it is",
        "let's": "let us",
        "ma'am": "madam",
        "mayn't": "may not",
        "might've": "might have",
        "mightn't": "might not",
        "must've": "must have",
        "mustn't": "must not",
        "needn't": "need not",
        "oughtn't": "ought not",
        "shan't": "shall not",
        "sha'n't": "shall not",
        "she'd": "she would",
        "she'll": "she will",
        "she's": "she is",
        "should've": "should have",
        "shouldn't": "should not",
        "that'd": "that would",
        "that's": "that is",
        "there'd": "there had",
        "there's": "there is",
        "they'd": "they would",
        "they'll": "they will",
        "they're": "they are",
        "they've": "they have",
        "wasn't": "was not",
        "we'd": "we would",
        "we'll": "we will",
        "we're": "we are",
        "we've": "we have",
        "weren't": "were not",
        "what'll": "what will",
        "what're": "what are",
        "what's": "what is",
        "what've": "what have",
        "where'd": "where did",
        "where's": "where is",
        "who'll": "who will",
        "who's": "who is",
        "won't": "will not",
        "wouldn't": "would not",
        "you'd": "you would",
        "you'll": "you will",
        "you're": "you are"
    }

    # Load data
    df_original = pd.read_json("Food.json")
    df_original = df_original[['overall', 'reviewText']]
    mask = df_original['overall'] != 3
    df = df_original[mask]
    # Splitting dataset into positive (4,5 stars) and negative (1-2 stars) reviwes
    df['Label'] = 0
    df.loc[df['overall'] > 3, ['Label']] = 1
    df = df.dropna()

    """
    # Plot data
    ax = df['overall'].value_counts().plot(kind='bar', figsize=(6, 6))
    fig = ax.get_figure()
    ax.set_title("Amazonz Reviews")
    ax.set_xlabel('Stars')
    ax.set_ylabel('Total');
    fig.show()
    """

    ngram_size = 1

    print('Ngram length: ' + str(ngram_size))
    avergage_accuracy = 0
    avergage_fscore = 0
    # Do 10 samples with each 10k reviews and only 1,2 and 4,5 stars reviews
    for i in range(10):
        df = df.sample(n=10000)
        # Clean reviews
        df['Cleaned_reviews'] = list(map(clean_reviews, df.reviewText))
        lemmatize_words(df.Cleaned_reviews)
        # Split dataset into train and test set
        training_data, test_data = sklearn.model_selection.train_test_split(df, train_size=0.8, random_state=42)
        y_train = training_data['Label']
        y_test = test_data['Label']
        # Create bag of words vector
        bow_transform = CountVectorizer(tokenizer=lambda doc: doc, ngram_range=[ngram_size, ngram_size], lowercase=False)
        x_train = bow_transform.fit_transform(training_data['Cleaned_reviews'])
        x_test = bow_transform.transform(test_data['Cleaned_reviews'])
        # Train model
        model = LogisticRegression(C=1).fit(x_train, y_train)
        # Print results
        accuracy = model.score(x_test, y_test)
        predict_test = model.predict(x_test)
        fscore = sklearn.metrics.f1_score(predict_test, y_test)
        print('Run', str(i+1), ': test prediction score (accuracy):', accuracy)
        print('Run', str(i + 1), ': test fscore:', fscore)
        avergage_accuracy += accuracy
        avergage_fscore += fscore

    avergage_accuracy /= 10
    avergage_fscore /= 10
    print('Total average test prediction score (accuracy):', avergage_accuracy)
    print('Total average test fscore:', avergage_fscore)

