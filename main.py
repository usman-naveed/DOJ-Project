# TODO: remove punctuation and special characters
from src.readData import ReadData
from src.preprocessData import PreProcess
import nltk
import pandas as pd
nltk.download('punkt')


def test():
    x = 5


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
    fDist.plot(20)
    FDist.plot(30)
    FDIST.plot(50)


if __name__ == '__main__':
    frequencyChartForTitles()
