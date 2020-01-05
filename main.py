from src.readData import ReadData
from src.preprocessData import PreProcess
import nltk
import pandas as pd

nltk.download('punkt')


def main():
    x = 7


def test():
    x = 5


def frequencyChartForTopics():
    # TODO: Topics are already in a list format, need to unlist then call tokenize method to avoid a nested list 3 levels deep.

    pd.set_option('display.width', 800)
    df = ReadData('/Users/musmannaveed/PycharmProjects/dojProject/data/combined.json').read_json()
    gh = df['title']
    df = df['topics']
    print(gh)
    print(df)
    for topic in df:
        if not topic:
            pass
        else:
            token = PreProcess(topic).tokenize()
    print(token)


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
    frequencyChartForTopics()
