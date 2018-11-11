import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import nltk

from plot_utils import plot_confusion_matrix, plot_keywords
from preprocessing_data import PreprocessingData


class Model(LogisticRegression):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vectorizer = None
        self.keywords = None

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
            word_coef = [(coef, nltk.pos_tag([index_to_word[i]])[0]) for i, coef in enumerate(self.coef_[class_index])]
            sorted_coef = sorted(word_coef, key=lambda x: x[0], reverse=True)

            top_nouns = [(coef, pos[0]) for coef, pos in sorted_coef if 'NN' in pos[1]][:n]
            top_adjectives = [(coef, pos[0]) for coef, pos in sorted_coef if 'JJ' in pos[1]][:n]
            top_verbs = [(coef, pos[0]) for coef, pos in sorted_coef if 'VB' in pos[1]][:n]

            top = sorted(sorted_coef[:n], key=lambda x: x[0])
            bottom = sorted_coef[-n:]

            keywords[class_index] = {
                'top': top,
                'bottom': bottom,
                'top_nouns': top_nouns,
                'top_adjectives': top_adjectives,
                'top_verbs': top_verbs
            }
        self.keywords = keywords
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
        if not self.keywords:
            keywords = self.get_keywords(n=10)
        else:
            keywords = self.keywords
        for genre in keywords:
            top = keywords[genre]['top']
            bottom = keywords[genre]['bottom']
            plot_keywords(top, bottom, self.classes_[genre])

    def keywords_table(self):
        """print data for latex table"""
        if not self.keywords:
            keywords = self.get_keywords(n=10)
        else:
            keywords = self.keywords
        for genre in keywords:
            top = sorted(keywords[genre]['top'], reverse=True)
            bottom = sorted(keywords[genre]['bottom'])
            print("\n", self.classes_[genre])
            print("\hline")
            print("+ Keywords & Weight & - Keywords & Weight \\\\")
            print("\hline")
            for el in zip(top, bottom):
                print("%s & %.2f & %s & %.2f \\\\" % (el[0][1][0], el[0][0], el[1][1][0], el[1][0]))

    def pos_keywords_table(self):
        """Print tagged keywords for latex table."""
        for genre in self.keywords:
            verbs = self.keywords[genre]['top_verbs']
            nouns = self.keywords[genre]['top_nouns']
            adjectives = self.keywords[genre]['top_adjectives']

            print(verbs)
            print(nouns)
            print(adjectives)

            print("\n", self.classes_[genre])
            print("\hline")
            print("Top verbs & Weight \\\\")
            print("\hline")
            for v in verbs:
                print("%s & %.2f \\\\" % (v[1], v[0]))

            print("\n", self.classes_[genre])
            print("\hline")
            print("Top nouns & Weight \\\\")
            print("\hline")
            for n in nouns:
                print("%s & %.2f \\\\" % (n[1], n[0]))

            print("\n", self.classes_[genre])
            print("\hline")
            print("Top adjectives & Weight \\\\")
            print("\hline")
            for a in adjectives:
                print("%s & %.2f \\\\" % (a[1], a[0]))

    def check_wrong_classified(self, df, y_test, y_predicted, y_probability, id_test):
        hhp = pd.DataFrame()
        pm = pd.DataFrame()
        hhp_coef_hip_hop = []
        hhp_coef_pop = []
        pm_coef_pop = []
        pm_coef_metal = []
        for i in range(len(y_test)):
            if y_test[i] == 'Hip-Hop' and y_predicted[i] == 'Pop':
                hhp = hhp.append(df.loc[[id_test[i]]])
                hhp_coef_hip_hop.append(y_probability[i][0])
                hhp_coef_pop.append(y_probability[i][2])

            if y_test[i] == 'Pop' and y_predicted[i] == 'Metal':
                pm = pm.append(df.loc[[id_test[i]]])
                pm_coef_pop.append(y_probability[i][2])
                pm_coef_metal.append(y_probability[i][1])

        hhp = hhp.drop(columns=['lyrics', 'word_count', 'text_lemmatized'])
        pm = pm.drop(columns=['lyrics', 'word_count', 'text_lemmatized'])

        hhp_diff = [y - x for x, y in zip(hhp_coef_hip_hop, hhp_coef_pop)]
        pm_diff = [y - x for x, y in zip(pm_coef_pop, pm_coef_metal)]

        se = pd.Series(hhp_coef_hip_hop)
        hhp['coeficient_hip_hop'] = se.values
        se = pd.Series(hhp_coef_pop)
        hhp['coeficient_pop'] = se.values
        se = pd.Series(hhp_diff)
        hhp['coeficient_diff'] = se.values
        hhp.sort_values("coeficient_diff", inplace=True)
        hhp_top = hhp.head(100)
        hhp_bottom = hhp.tail(100)
        hhp_new = hhp_top.append(hhp_bottom)

        hhp_new.to_csv("hip-hop_pop.csv")

        se = pd.Series(pm_coef_pop)
        pm['coeficient_pop'] = se.values
        se = pd.Series(pm_coef_metal)
        pm['coeficient_metal'] = se.values
        se = pd.Series(pm_diff)
        pm['coeficient_diff'] = se.values
        pm.sort_values("coeficient_diff", inplace=True)
        pm_top = pm.head(100)
        pm_bottom = pm.tail(100)
        pm_new = pm_top.append(pm_bottom)

        pm_new.to_csv("pop_metal.csv")

    def test_model(self, data, labels, indices, n=1, seed=None):
        """Print average metrics over n iterations."""
        a_sum, p_sum, r_sum, f_sum = 0, 0, 0, 0

        for i in range(n):
            # train model
            X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(data, labels, indices, test_size=0.2, random_state=seed)
            X_train = self.vectorize_data(X_train)
            model.fit(X_train, y_train)

            # predict genres
            X_test = self.vectorizer.transform(X_test)
            y_predicted = self.predict(X_test)
            y_probability = model.predict_proba(X_test)

            a, p, r, f = self.get_metrics(y_test, y_predicted)
            a_sum += a
            p_sum += p
            r_sum += r
            f_sum += f

        print('Average over %d iterations: acc = %.3f, prec = %.3f, rec = %.3f, f1 = %.3f' %
              (n, a_sum/n, p_sum/n, r_sum/n, f_sum/n))

        return y_test, y_predicted, y_probability, id_test


if __name__ == '__main__':
    # model init
    model = Model(C=1e-1, class_weight='balanced', solver='lbfgs', multi_class='multinomial', max_iter=200)
    df = pd.read_csv('./input/lyrics_clean_2015_2016.csv')

    list_genres = df['genre'].tolist()
    list_lyrics = df['lyrics'].tolist()
    idx = list(range(len(list_genres)))

    # model testing
    # model.test_model(list_lyrics, list_genres, n=10)

    # Final model with predefined seed (for results replication)
    y_test, y_predicted, y_probability, id_test = model.test_model(list_lyrics, list_genres, idx, n=1, seed=123)

    # plot model
    model.plot_confusion_matrix(y_test, y_predicted)
    model.plot_keywords()

    model.check_wrong_classified(df, y_test, y_predicted, y_probability, id_test)

    # Helpers for latex tables
    # model.keywords_table()
    # model.pos_keywords_table()
