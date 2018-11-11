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
            word_coef = [(coef, index_to_word[i]) for i, coef in enumerate(self.coef_[class_index])]
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
        plot_confusion_matrix(cm, classes=self.classes_, normalize=False)

    def plot_keywords(self):
        keywords = self.get_keywords(n=10)
        for genre in keywords:
            top = keywords[genre]['top']
            bottom = keywords[genre]['bottom']
            plot_keywords(top, bottom, self.classes_[genre])

    def keywords_table(self):
        """print data for latex table"""
        keywords = self.get_keywords(n=10)
        for genre in keywords:
            top = sorted(keywords[genre]['top'], reverse=True)
            bottom = sorted(keywords[genre]['bottom'])
            print("\n", self.classes_[genre])
            print("\hline")
            print("+ Keywords & Weight & - Keywords & Weight \\\\")
            print("\hline")
            for el in zip(top, bottom):
                print("%s & %.2f & %s & %.2f \\\\" % (el[0][1], el[0][0], el[1][1], el[1][0]))

    def test_model(self, data, labels, n=1, seed=None):
        """Print average metrics over n iterations."""
        a_sum, p_sum, r_sum, f_sum = 0, 0, 0, 0

        for i in range(n):
            # train model
            X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=seed)
            X_train = self.vectorize_data(X_train)
            model.fit(X_train, y_train)

            # predict genres
            X_test = self.vectorizer.transform(X_test)
            y_predicted = self.predict(X_test)

            a, p, r, f = self.get_metrics(y_test, y_predicted)
            a_sum += a
            p_sum += p
            r_sum += r
            f_sum += f

        print('Average over %d iterations: acc = %.3f, prec = %.3f, rec = %.3f, f1 = %.3f' %
              (n, a_sum/n, p_sum/n, r_sum/n, f_sum/n))

        return y_test, y_predicted


if __name__ == '__main__':
    # model init
    model = Model(C=1e-1, class_weight='balanced', solver='lbfgs', multi_class='multinomial', max_iter=200)
    df = pd.read_csv('./input/lyrics_clean_2014.csv')
    #pp = PreprocessingData(data='./input/lyrics.csv')
    #df = pp.clean_data()

    list_genres = df['genre'].tolist()
    list_lyrics = df['lyrics'].tolist()

    y_test, y_predicted = model.test_model(list_lyrics, list_genres, n=2)

    # plot model
    model.plot_confusion_matrix(y_test, y_predicted)
    model.plot_keywords()
    model.keywords_table()
