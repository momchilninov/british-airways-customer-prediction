import os
import pandas as pd
import numpy as np
import random
import textacy.preprocessing as tprep #Character Normalization 
import textacy.extract as txtrct
import regex as re
import nltk
import matplotlib.pyplot as plt
from collections import Counter # bag of words for topic modelling
from wordcloud import WordCloud # Creating Word Clouds (from frequency diagrams)
from matplotlib import pyplot as plt
from collections import Counter
from textacy.extract import keyword_in_context as KWIC #Finding a Keyword-in-Context


PATH = "data/raw/BA_reviews.csv"

def read_csv(relative_path):
    # Get the directory of the current script
    script_directory = os.path.dirname(os.path.abspath(__file__))

    # Construct the relative path to the data directory
    data_directory = os.path.join(script_directory, "../..", relative_path)

    df = pd.read_csv(data_directory)
    
    columns_to_drop = [col for col in df.columns if 'Unnamed' in col]
    df.drop(columns=columns_to_drop, inplace=True)

    df = df.copy()
    return df

# Call the function to read the CSV files
BA_reviews_df = read_csv(PATH)


BA_reviews_df["reviews"] = BA_reviews_df["reviews"].str.replace(r'^.*?\w*\s*\|', '', regex=True) #matches everything from the start of the line up to and including the first occurrence of "|" (including the "|" character itself)

BA_reviews_df["reviews"] = BA_reviews_df["reviews"].str.strip()

BA_reviews_df["reviews"] = BA_reviews_df["reviews"].str.replace("-", " ")


## Function: Character Normalization 
def normalize(text):
    text = tprep.normalize.hyphenated_words(text)
    text = tprep.normalize.quotation_marks(text)
    text = tprep.normalize.unicode(text)
    text = tprep.remove.accents(text)
    return text

## Function: tokenization
def tokenize(text):
    return re.findall(r'[\w-]*\p{L}[\w-]*', text)


## Function: remove stop words
stopwords = set(nltk.corpus.stopwords.words('english'))

include_stopwords = {'us', "would"}
exclude_stopwords = {'not'}

stopwords |= include_stopwords
stopwords -= exclude_stopwords

def remove_stop(tokens):
    return [t for t in tokens if t.lower() not in stopwords]

### Function: create pipeline
pipeline = [normalize, str.lower, tokenize, remove_stop]

def prepare(text, pipeline):
    tokens = text
    for transform in pipeline:
        tokens = transform(tokens)
    return tokens

BA_reviews_df["tokens"] = BA_reviews_df["reviews"].apply(prepare, pipeline=pipeline)

BA_reviews_df["num_tokens"] = BA_reviews_df["tokens"].map(len)

# bag of words for topic modelling
counter = Counter()
BA_reviews_df["tokens"].map(counter.update)


def count_words(df, column='tokens', preprocess=None, min_freq=2):
    # process tokens and update counter
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(tokens)
    # create counter and run through all data
    counter = Counter()
    df[column].map(update)
    # transform counter into a DataFrame
    freq_df = pd.DataFrame.from_dict(counter, orient='index', columns=['freq'])
    freq_df = freq_df.query('freq >= @min_freq')
    freq_df.index.name = 'token'
    return freq_df.sort_values('freq', ascending=False)

freq_df = count_words(BA_reviews_df)

# Creating Word Clouds (from frequency diagrams)
def wordcloud(word_freq, title=None, max_words=200, stopwords=None, save_path=None, figure_name=None):
    wc = WordCloud(width=800, height=400,
                   background_color= "black", colormap="Paired",
                   max_font_size=150, max_words=max_words)
    # convert DataFrame into dict
    if type(word_freq) == pd.Series:
        counter = Counter(word_freq.fillna(0).to_dict())
    else:
        counter = word_freq
    # filter stop words in frequency counter
    if stopwords is not None:
        counter = {token:freq for (token, freq) in counter.items() if token not in stopwords}
    wc.generate_from_frequencies(counter)
    plt.title(title)
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    
    # Check if save_path and figure_name are provided, then save the word cloud
    if save_path and figure_name:
        plt.savefig(f"{save_path}/{figure_name}", bbox_inches='tight', pad_inches=0.1, dpi=300)


## Ranking with TF-IDF
#compute the IDF for all terms in the corpus
def compute_idf(df, column='tokens', preprocess=None, min_df=2):
    def update(doc):
        tokens = doc if preprocess is None else preprocess(doc)
        counter.update(set(tokens))
    # count tokens
    counter = Counter()
    df[column].map(update)
    # transform counter into a DataFrame
    idf_df = pd.DataFrame.from_dict(counter, orient='index', columns=['df'])
    idf_df = idf_df.query('df >= @min_df')
    idf_values = np.log(len(df)/(idf_df['df']))+0.1
    idf_df.index.name = 'token'
    
    # Create a new DataFrame with the calculated IDF values
    new_idf_df = idf_df.copy()
    new_idf_df['idf'] = idf_values
    return new_idf_df


# IDF values
idf_df = (compute_idf(BA_reviews_df))


# calculate the TF-IDF score for the terms
freq_df['tfidf'] = freq_df["freq"] * idf_df['idf']


## Function: Finding a Keyword-in-Context
# we see that not appears quite often 
def kwic(doc_series, keyword, window=35, print_samples=10):
    def add_kwic(text):
        kwic_list.extend(KWIC(text, keyword, ignore_case=True,
                              window_width=window))
    kwic_list = []
    doc_series.map(add_kwic)
    
    if print_samples is None or print_samples==0:
        return kwic_list
    else:
        k = min(print_samples, len(kwic_list))
        print(f"{k} random samples out of {len(kwic_list)} " + \
            f"contexts for '{keyword}':")
        for sample in random.sample(list(kwic_list), k):
            print(re.sub(r'[\n\t]', ' ', sample[0])+' '+ \
                sample[1]+' '+\
                    re.sub(r'[\n\t]', ' ', sample[2]))


kwic(BA_reviews_df['reviews'], "not")
kwic(BA_reviews_df['reviews'], "seat")


## Analyzing N-Grams
# def ngrams(tokens, n=2, sep=' '):
#     return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)])]

# text = "the visible manifestation of the global climate change"
# tokens = tokenize(text)
# print("|".join(ngrams(tokens, 2)))

def ngrams(tokens, n=2, sep=' ', stopwords=set()):
    return [sep.join(ngram) for ngram in zip(*[tokens[i:] for i in range(n)]) 
            
            if len([t for t in ngram if t in stopwords])==0]


# add a column containing all bigrams to our df
BA_reviews_df['bigrams'] = BA_reviews_df['reviews'].apply(prepare, 
                                                          pipeline=[str.lower,tokenize]) \
.apply(ngrams, n=2, stopwords=stopwords)


count_words(BA_reviews_df, 'bigrams').head(5)

#### frequency df for bigrams
freq_df_bigrams = count_words(BA_reviews_df, column="bigrams")

#### idf for bigrams 
bigram_idf = compute_idf(BA_reviews_df, 'bigrams', min_df=10)

####frequency df only for bigrams 
freq_df_bigrams['tfidf'] = freq_df_bigrams['freq'] * bigram_idf['idf']

# frequency df that includes unigrams and bigrams 
freq_df_updated = pd.concat([freq_df, freq_df_bigrams], axis=0)



