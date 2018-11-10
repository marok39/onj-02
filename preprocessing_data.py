import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from langdetect import detect


class PreprocessingData:
    def __init__(self, data=None):
        self.data = data
        self.w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(self, text):
        stop = stopwords.words('english')
        return [self.lemmatizer.lemmatize(w) for w in self.w_tokenizer.tokenize(text) if w not in stop]

    def clean_data(self):
        if not self.data:
            raise ValueError('PreprocessingData.data not defined.')

        # import data
        df = pd.read_csv(self.data)

        # replace new lines
        df = df.replace({'\n': ' '}, regex=True)

        # get word count for songs
        df['word_count'] = df['lyrics'].str.split().str.len()

        # select only Hip-Hop, Pop & Rock genres
        df = df.loc[df['genre'].isin(['Hip-Hop', 'Pop', 'Metal'])]
        # select song from 2015 onward
        df = df.loc[df['year'] >= 2015]

        # plot words per song
        # sns.violinplot(x=df["word_count"])
        # plt.show()

        # use only song with more than 100 words and less than 1000 because of outliers
        df = df[df['word_count'] >= 100]
        df_clean = df[df['word_count'] <= 1000]
        print(df_clean['word_count'].groupby(df_clean['genre']).describe())

        # count song by genre
        genre = df_clean.groupby(['genre'], as_index=False).count()
        genre2 = genre[['genre', 'song']]
        print(genre2)

        # again plot words per song
        # sns.violinplot(x=df_clean["word_count"])
        # plt.show()

        # remove brackets with text (Ex. [Verse 1])
        df_new = df_clean.replace({'([\[]).*?([\]])': ''}, regex=True)
        # change all text to lower case
        df_new['lyrics'] = df_new['lyrics'].str.lower()
        # remove punctuations
        df_new['lyrics'] = df_new['lyrics'].str.replace('[^\w\s^\']', '')

        # select only english lyrics
        # TODO: vcasih vrze langdetect.lang_detect_exception.LangDetectException: No features in text., vrjetn bo treba for pa catch exception
        df_new['language'] = df_new['lyrics'].apply(lambda x: detect(x[:100]))
        df_new = df_new.loc[df_new['language'] == 'en']
        """
        # lemmatization of text
        df_new['text_lemmatized'] = df_new.lyrics.apply(self.lemmatize_text)

        # remove english stop words
        stop = stopwords.words('english')
        df_new['text_lemmatized_without_stop_words'] = df_new['text_lemmatized'].apply(lambda x: [item for item in x if item not in stop])
        """
        df_new.to_csv('./input/lyrics_clean_2015_2016.csv')
        # df_new.to_csv('./input/lyrics_test.csv')
        return df_new


if __name__ == '__main__':
    pp = PreprocessingData('./input/lyrics.csv')
    df = pp.clean_data()

