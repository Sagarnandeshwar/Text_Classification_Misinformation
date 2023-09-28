import nltk
import numpy as np
import string

from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

count_vect = CountVectorizer()
tfidf_transformer = TfidfTransformer()

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
stopwords = stopwords.words('english')

model1 = GaussianNB()
model2 = LogisticRegression()
model3 = SVC()

# Best Hyperparameter Model
"""
model1 = GaussianNB(var_smoothing=1e-07)
model2 = LogisticRegression(C=10)
model3 = SVC(C=1, gamma=1, kernel='linear')
"""

test_dataset_size = 0.25


def load_data(x_list, y_list, train_file_path, fact):
    dataset = open(train_file_path).readlines()
    if fact:
        for lines in dataset:
            x_list.append(lines)
            y_list.append("fact")
    else:
        for lines in dataset:
            x_list.append(lines)
            y_list.append("fake")
    return x_list, y_list


def feature_extraction_tokenization(train_list):
    new_list = []
    for line in train_list:
        new_list.append(word_tokenize(line.lower()))
    return new_list


def feature_extraction_detokenization(train_list):
    new_list = []
    for line in train_list:
        new_list.append(TreebankWordDetokenizer().detokenize(line))
    return new_list


def feature_extraction_remove_stopword(train_list):
    new_list = []
    for line in train_list:
        new_line = [token for token in line if token not in stopwords]
        new_list.append(new_line)
    return new_list


def feature_extraction_remove_punctuation(train_list):
    new_list = []
    for line in train_list:
        new_line = [token for token in line if token not in string.punctuation]
        new_list.append(new_line)
    return new_list


def feature_extraction_lemmatization(train_list):
    new_list = []
    for line in train_list:
        new_line = [lemmatizer.lemmatize(token) for token in line]
        new_list.append(new_line)
    return new_list


def feature_extraction_stemming(train_list):
    new_list = []
    for line in train_list:
        new_line = [stemmer.stem(token) for token in line]
        new_list.append(new_line)
    return new_list


if __name__ == '__main__':

    # Loading Datasets
    X, y = [], []
    X, y = load_data(X, y, "facts.txt", True)
    X, y = load_data(X, y, "fakes.txt", False)

    # Split data into Train and Test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, test_size=test_dataset_size, shuffle=True)

    # Pre-processing
    X_train = feature_extraction_tokenization(X_train)
    X_train = feature_extraction_lemmatization(X_train)
    X_train = feature_extraction_stemming(X_train)
    X_train = feature_extraction_remove_stopword(X_train)
    X_train = feature_extraction_remove_punctuation(X_train)
    X_train = feature_extraction_detokenization(X_train)

    # Count Vectorization and TF-IDF
    X_train_counts = count_vect.fit_transform(X_train)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

    """
    # Hyperparameter tuning
    param_model1 = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]}
    param_model2 = {
        'solver': ['lbfgs', 'sag', 'liblinear', 'newton-cg'],
        'penalty': ['l1', 'l2', 'elasticnet', 'None'],
        'C': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    }
    param_model3 = {
        'C': [0.1, 1, 10, 100],
        'gamma': [1, 0.1, 0.01, 0.001],
        'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
    }

    model1_grid = GridSearchCV(estimator=model1, param_grid=param_model1, verbose=1, cv=10, n_jobs=-1)
    model2_grid = GridSearchCV(estimator=model2, param_grid=param_model2, verbose=1, cv=10, n_jobs=-1)
    model3_grid = GridSearchCV(estimator=model3, param_grid=param_model3, verbose=1, cv=10, n_jobs=-1)

    model1_grid.fit(X_train_tfidf.toarray(), y_train)
    model2_grid.fit(X_train_tfidf.toarray(), y_train)
    model3_grid.fit(X_train_tfidf.toarray(), y_train)

    print(model1_grid.best_estimator_)
    print(model2_grid.best_estimator_)
    print(model3_grid.best_estimator_)
    """

    # Training
    model1.fit(X_train_tfidf.toarray(), y_train)
    model2.fit(X_train_tfidf.toarray(), y_train)
    model3.fit(X_train_tfidf.toarray(), y_train)

    # Prediction
    """
    X_test = feature_extraction_tokenization(X_test)
    X_test = feature_extraction_remove_stopword(X_test)
    X_test = feature_extraction_remove_punctuation(X_test)
    X_test = feature_extraction_lemmatization(X_test)
    X_test = feature_extraction_stemming(X_test)
    X_test = feature_extraction_detokenization(X_test)
    """

    X_test_counts = count_vect.transform(X_test)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)

    # Model 1
    y_cap_test_model1 = model1.predict(X_test_tfidf.toarray())
    y_cap_test_model2 = model2.predict(X_test_tfidf.toarray())
    y_cap_test_model3 = model3.predict(X_test_tfidf.toarray())

    # Evaluation
    # Model 1
    print("Model 1 : Naive Bayes")
    print(accuracy_score(y_cap_test_model1, y_test))
    print(f1_score(y_cap_test_model1, y_test, average='macro'))

    # Model 2
    print("Model 2 : Logistic Regression")
    print(accuracy_score(y_cap_test_model2, y_test))
    print(f1_score(y_cap_test_model2, y_test, average='macro'))

    # Model 3
    print("Model 3 : Support Vector Machines")
    print(accuracy_score(y_cap_test_model3, y_test))
    print(f1_score(y_cap_test_model3, y_test, average='macro'))

