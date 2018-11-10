import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

from plot_utils import plot_confusion_matrix, plot_keywords
from preprocessing_data import PreprocessingData


class Model(LogisticRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = None

    def vectorize_data(self, data):
        """Vectorizes data with TF-IDF."""
        pp = PreprocessingData()
        self.vectorizer = TfidfVectorizer(tokenizer=pp.lemmatize_text)
        return self.vectorizer.fit_transform(data)

    def get_keywords(self, n=10):
        """Gets top n model weights with highest positive and negative value."""
        if not self.vectorizer:
            raise ValueError('Vectorizer is not defined! Vectorize train data first using Model.vectorize_data.')

        index_to_word = {v: k for k, v in self.vectorizer.vocabulary_.items()}

        keywords = {}
        for class_index in range(self.coef_.shape[0]):
            word_coef = [(word, index_to_word[i]) for i, word in enumerate(self.coef_[class_index])]
            sorted_coef = sorted(word_coef, key=lambda x: x[0], reverse=True)
            top = sorted(sorted_coef[:n], key=lambda x: x[0])
            bottom = sorted_coef[-n:]
            keywords[class_index] = {
                'top': top,
                'bottom': bottom
            }
        return keywords

    @staticmethod
    def get_metrics(y_test, y_predicted):
        """Get model metrics: precision, recall, F1-score, accuracy."""
        # true positives / (true positives+false positives)
        prec = precision_score(y_test, y_predicted, pos_label=None, average='weighted')

        # true positives / (true positives + false negatives)
        rec = recall_score(y_test, y_predicted, pos_label=None, average='weighted')

        # harmonic mean of precision and recall
        f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

        # true positives + true negatives/ total
        acc = accuracy_score(y_test, y_predicted)

        print('accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f' % (acc, prec, rec, f1))

        return acc, prec, rec, f1

    def plot_confusion_matrix(self, y_test, y_predicted):
        cm = confusion_matrix(y_test, y_predicted)
        plot = plot_confusion_matrix(cm, classes=self.classes_, normalize=False)

    def plot_keywords(self):
        keywords = self.get_keywords(n=10)
        for genre in keywords:
            top = keywords[genre]['top']
            bottom = keywords[genre]['bottom']
            plot_keywords(top, bottom, self.classes_[genre])


if __name__ == '__main__':
    # model init
    model = Model(C=1.0, class_weight='balanced', solver='lbfgs', multi_class='multinomial', max_iter=200)  # solver='newton-cg'
    df = pd.read_csv('./input/lyrics_clean_2015_2016.csv')
    #pp = PreprocessingData(data='./input/lyrics.csv')
    #df = pp.clean_data()

    list_genres = df['genre'].tolist()
    list_lyrics = df['lyrics'].tolist()

    # train model
    X_train, X_test, y_train, y_test = train_test_split(list_lyrics, list_genres, test_size=0.2, random_state=123)
    X_train = model.vectorize_data(X_train)
    model.fit(X_train, y_train)

    # predict genres
    X_test = model.vectorizer.transform(X_test)
    y_predicted = model.predict(X_test)

    # plot model
    model.get_metrics(y_test, y_predicted)
    model.plot_confusion_matrix(y_test, y_predicted)
    model.plot_keywords()
