from src.readData import ReadData
from src.preprocessData import PreProcess
import nltk
from itertools import chain
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

nltk.download('punkt')


# This method creates a frequency chart based on the topics of DOJ documents
def frequency_chart_for_topics():
    pd.set_option('display.width', 800)
    df = ReadData("data/combined.json").read_json()
    df = list(chain.from_iterable(df['topics']))
    f_dist = nltk.probability.FreqDist(df)
    f_dist.tabulate(title='Frequency Count of Topics')


# This method creates a frequency chart based on the titles of DOJ documents
def frequency_chart_for_titles():
    pd.set_option('display.width', 800)
    df = ReadData('data/combined.json').read_json()
    tokenized_titles = PreProcess(df['title']).reg_exp_tokenize()
    tokenized_titles_stop_word_removed = PreProcess.remove_stop_words(tokenized_titles, return_type='tokens')
    bigrams_stop_words_removed = nltk.bigrams(tokenized_titles_stop_word_removed)
    trigrams_stop_words_removed = nltk.trigrams(tokenized_titles_stop_word_removed)
    f_dist = nltk.probability.FreqDist(tokenized_titles_stop_word_removed)
    bigrams = nltk.probability.FreqDist(bigrams_stop_words_removed)
    trigrams = nltk.probability.FreqDist(trigrams_stop_words_removed)
    f_dist.plot(20, title='Frequency Chart of Words appearing in the Title')
    bigrams.plot(30, title='Frequency Chart of Bigrams appearing in the Title')
    trigrams.plot(50, title='Frequency Chart of Trigrams appearing in the Title')


# This method creates a word cloud from the contents feature in the data
def word_cloud_for_content():
    pd.set_option('display.width', 800)
    df = ReadData('data/combined.json').read_json()
    tokenized_content = PreProcess(df['contents']).reg_exp_tokenize()
    tokenized_content_stop_words_removed = PreProcess.remove_stop_words(tokenized_content, return_type='tokens')
    contents_string = " ".join(content for content in tokenized_content_stop_words_removed)
    wordcloud = WordCloud(background_color="white").generate(contents_string)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

