from src.readData import ReadData
from src.preprocessData import PreProcess
import nltk
from itertools import chain
import pandas as pd

nltk.download('punkt')


def main():
    x = 7


def test():
    x = 5


def frequencyChartForTopics():
    pd.set_option('display.width', 800)
    df = ReadData('/Users/musmannaveed/PycharmProjects/dojProject/data/combined.json').read_json()
    df = list(chain.from_iterable(df['topics']))
    fDist = nltk.probability.FreqDist(df)
    fDist.plot(title='Frequency Count of Topics')


def frequencyChartForTitles():
    pd.set_option('display.width', 800)
    df = ReadData('/Users/musmannaveed/PycharmProjects/dojProject/data/combined.json').read_json()
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


if __name__ == '__main__':
    frequencyChartForTopics()
