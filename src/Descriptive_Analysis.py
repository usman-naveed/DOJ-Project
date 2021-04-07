from src.readData import ReadData
from src.preprocessData import PreProcess
import nltk
from itertools import chain
import pandas as pd

nltk.download('punkt')


# This method creates a frequency chart based on the topics of DOJ documents
def frequencyChartForTopics():
    pd.set_option('display.width', 800)
    df = ReadData("/home/yungest/PycharmProjects/DOJ-Project/data/combined.json").read_json()
    df = list(chain.from_iterable(df['topics']))
    fDist = nltk.probability.FreqDist(df)
    fDist.plot(title='Frequency Count of Topics')


# This method creates a frequency chart based on the titles of DOJ documents
def frequencyChartForTitles():
    pd.set_option('display.width', 800)
    df = ReadData('/home/yungest/PycharmProjects/DOJ-Project/data/combined.json').read_json()
    tokenizedTitles = PreProcess(df['title']).regExpTokenize()
    tokenizedTitlesStopWordRemoved = PreProcess.removeStopWords(tokenizedTitles, returnType='tokens')
    bigramsStopWordsRemoved = nltk.bigrams(tokenizedTitlesStopWordRemoved)
    trigramsStopWordsRemoved = nltk.trigrams(tokenizedTitlesStopWordRemoved)
    fDist = nltk.probability.FreqDist(tokenizedTitlesStopWordRemoved)
    FDist = nltk.probability.FreqDist(bigramsStopWordsRemoved)
    FDIST = nltk.probability.FreqDist(trigramsStopWordsRemoved)
    fDist.plot(20, title='Frequency Chart of Words appearing in the Title')
    FDist.plot(30, title='Frequency Chart of Bigrams appearing in the Title')
    FDIST.plot(50, title='Frequency Chart of Trigrams appearing in the Title')
