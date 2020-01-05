import nltk
nltk.download('stopwords')


class PreProcess:
    def __init__(self, df):
        self.dataframe = df

    def tokenize(self):
        sentences = self.dataframe
        tokens = []
        for words in sentences:
            token = nltk.tokenize.word_tokenize(words)
            tokens.append(token)
        return tokens

    def regExpTokenize(self):
        """
        regExpTokenize removes punctuations, only keeps alpha-numeric tokens.
        """
        sentences = self.dataframe
        tokens = []
        for words in sentences:
            tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
            token = tokenizer.tokenize(words)
            tokens.append(token)
        return tokens

    @staticmethod
    def removeStopWords(tokenized, returnType: str = 'list'):
        # TODO: Add exceptions for argument type
        """
        The method removeStopWords takes an argument 'tokenized' which is assumed to be of the format [[],[],...,[]]
        """
        stopWords = set(nltk.corpus.stopwords.words('english'))
        filteredSentence = []
        if returnType == 'list':
            for sentence in tokenized:
                placeHolder = []
                for word in sentence:
                    if word not in stopWords:
                        placeHolder.append(word)
                filteredSentence.append(placeHolder)
            return filteredSentence
        elif returnType == 'tokens':
            for sentence in tokenized:
                for word in sentence:
                    if word not in stopWords:
                        filteredSentence.append(word)
            return filteredSentence
