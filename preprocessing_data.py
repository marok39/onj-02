import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from langdetect import detect

PRESELECTED_GENRES = ['Hip-Hop', 'Pop', 'Metal']

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
        print("importing data")
        df = pd.read_csv(self.data)

        # REDUCE DATA FRAME SIZE
        print("getting songs from 2015 onward")
        # select song from 2015 onward
        df = df.loc[df['year'] >= 2015]
        # select only genres defined in PRESELECTED_GENRES
        df = df.loc[df['genre'].isin(PRESELECTED_GENRES)]

        # replace new lines
        df = df.replace({'\n': ' '}, regex=True)

        # get word count for songs
        df['word_count'] = df['lyrics'].str.split().str.len()

        # plot words per song
        # sns.violinplot(x=df["word_count"])
        # plt.show()

        # use only song with more than 100 words and less than 1000 because of outliers
        print("selecting songs with 100-1000 words")
        df = df[df['word_count'] >= 100]
        df_clean = df[df['word_count'] <= 1000]
        print(df_clean['word_count'].groupby(df_clean['genre']).describe())

        # count song by genre
        print("counting songs per genre")
        genre = df_clean.groupby(['genre'], as_index=False).count()
        genre2 = genre[['genre', 'song']]
        print(genre2)

        # again plot words per song
        # sns.violinplot(x=df_clean["word_count"])
        # plt.show()

        print("cleaning text")
        # remove brackets with text (Ex. [Verse 1])
        df_new = df_clean.replace({'([\[]).*?([\]])': ''}, regex=True)
        # change all text to lower case
        df_new['lyrics'] = df_new['lyrics'].str.lower()
        # remove punctuations
        df_new['lyrics'] = df_new['lyrics'].str.replace('[^\w\s^\']', '')

        # select only english lyrics
        print("selecting english songs (needs ~10 seconds)")
        for row in df_new.itertuples(index=True, name='Pandas'):
            try:
                # if detect(getattr(row, "lyrics")[:200]) != 'en' or detect(getattr(row, "lyrics")[200:400]) != 'en':
                if detect(getattr(row, "lyrics")[:400]) != 'en':
                    df_new.drop(getattr(row, "index"), inplace=True)
            except:
                df_new.drop(getattr(row, "index"), inplace=True)

        # df_new['language'] = df_new['lyrics'].apply(lambda x: detect(x[:100]))
        # df_new = df_new.loc[df_new['language'] == 'en']

        # lemmatization of text & remove stop words
        print("lemmatization of text & remove stop words")
        df_new['text_lemmatized'] = df_new.lyrics.apply(self.lemmatize_text)
        """
        # remove english stop words
        stop = stopwords.words('english')
        df_new['text_lemmatized_without_stop_words'] = df_new['text_lemmatized'].apply(lambda x: [item for item in x if item not in stop])
        """
        print("saving processed data to file")
        df_new.to_csv('./input/lyrics_clean_2015_2016.csv')
        # df_new.to_csv('./input/lyrics_test.csv')
        print("done")
        return df_new


if __name__ == '__main__':
    pp = PreprocessingData('./input/lyrics.csv')
    df = pp.clean_data()

