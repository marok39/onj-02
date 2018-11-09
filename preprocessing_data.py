import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# import data
df = pd.read_csv('./input/lyrics.csv')

# replace new lines
df = df.replace({'\n': ' '}, regex=True)

# get word count for songs
df['word_count'] = df['lyrics'].str.split().str.len()

# select only Hip-Hop, Pop & Rock genres
df = df.loc[df['genre'].isin(['Hip-Hop', 'Pop', 'Rock'])]

# plot words per song
sns.violinplot(x=df["word_count"])
plt.show()

# use only song with more than 100 words and less than 1000 because of outliers
df = df[df['word_count'] >= 100]
df_clean = df[df['word_count'] <= 1000]
print(df_clean['word_count'].groupby(df_clean['genre']).describe())

# count song by genre
genre = df_clean.groupby(['genre'], as_index=False).count()
genre2 = genre[['genre', 'song']]
print(genre2)

# again plot words per song
sns.violinplot(x=df_clean["word_count"])
plt.show()

# remove brackets with text (Ex. [Verse 1])
df_new = df_clean.replace({'([\[]).*?([\]])': ''}, regex=True)
# change all text to lower case
df_new['lyrics'] = df_new['lyrics'].str.lower()
# remove punctuations
df_new['lyrics'] = df_new['lyrics'].str.replace('[^\w\s]', '')

df_new.to_csv('./input/lyrics_clean.csv')
