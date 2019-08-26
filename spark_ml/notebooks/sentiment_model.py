import re

from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def preprocessor(text):
    return re.sub("[^a-zA-Z]", " ", text)


def tokenizer(text):
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in word_tokenize(text) if len(t) > 3]


vectorizer = TfidfVectorizer(preprocessor=preprocessor, tokenizer=tokenizer)

pipeline = Pipeline([
    ('tfidf', vectorizer),
    ('clf', RandomForestClassifier()),
])

parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'clf__n_estimators': (40, 60, 100),
    'clf__max_depth': (5, 8)}
