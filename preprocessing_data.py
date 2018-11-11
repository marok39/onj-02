import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from langdetect import detect

PRESELECTED_GENRES = ['Hip-Hop', 'Pop', 'Metal']
BLACKLIST_GENRES = ['Other', 'Electronic', 'Not Available']
YEAR = 2000


class PreprocessingData:
    def __init__(self, data=None):
        self.data = data
        self.w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

    def lemmatize_text(self, text):
        """lemmatization of text & remove stop words"""
        stop = stopwords.words('english')
        return [self.lemmatizer.lemmatize(w) for w in self.w_tokenizer.tokenize(text) if w not in stop]

    def clean_data(self):
        if not self.data:
            raise ValueError('PreprocessingData.data not defined.')

        # import data
        print("Importing data...")
        df = pd.read_csv(self.data)

        # REDUCE DATA FRAME SIZE
        print("Getting songs from %d onward..." % YEAR)
        # select song from YEAR onward
        df = df.loc[df['year'] >= YEAR]
        # Remove Other and Not available
        df = df.loc[~df['genre'].isin(BLACKLIST_GENRES)]
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
        print("Selecting songs with 100-1000 words...")
        df = df[df['word_count'] >= 100]
        df_clean = df[df['word_count'] <= 1000]

        # again plot words per song
        # sns.violinplot(x=df_clean["word_count"])
        # plt.show()

        print("Cleaning text...")
        # remove brackets with text (Ex. [Verse 1])
        df_new = df_clean.replace({'([\[]).*?([\]])': ''}, regex=True)
        # change all text to lower case
        df_new['lyrics'] = df_new['lyrics'].str.lower()
        # remove punctuations
        df_new['lyrics'] = df_new['lyrics'].str.replace('[^\w\s^\']', '')
        df_new['lyrics'] = df_new['lyrics'].astype('U')

        # select only english lyrics
        print("Filtering non-english songs (be patient)...")
        index_to_drop = []
        for i, row in enumerate(df_new.itertuples(index=True, name='Pandas')):
            try:
                # if detect(getattr(row, "lyrics")[:200]) != 'en' or detect(getattr(row, "lyrics")[200:400]) != 'en':
                if detect(getattr(row, "lyrics")[:200]) != 'en':
                    index_to_drop.append(i)
            except:
                index_to_drop.append(i)

        # drop non-english songs
        df_new = df_new.drop(df_new.index[index_to_drop])

        # Print data results
        print("\n----------------------------------------------------\n")
        print(df_new['word_count'].groupby(df_new['genre']).describe())
        print("\n----------------------------------------------------\n")

        # count song by genre
        print("Counting songs per genre...")
        print("\n----------------------------------------------------\n")
        genre = df_new.groupby(['genre'], as_index=False).count()
        print(genre[['genre', 'song']])
        print("\n----------------------------------------------------\n")
        print("Number of songs after cleanup:", len(df_new.index))
        print("\n----------------------------------------------------\n")

        # print("Saving processed data to file...")
        csv_name = './input/lyrics_clean_' + str(YEAR) + '.csv'
        df_new.to_csv(csv_name)
        # df_new.to_csv('./input/lyrics_test.csv')

        print("Done.")
        return df_new


if __name__ == '__main__':
    pp = PreprocessingData('./input/lyrics.csv')
    df = pp.clean_data()
