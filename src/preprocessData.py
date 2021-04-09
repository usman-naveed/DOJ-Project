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

    def reg_exp_tokenize(self):
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
    def remove_stop_words(tokenized, return_type: str = 'list'):
        # TODO: Add exceptions for argument type
        """
        The method removeStopWords takes an argument 'tokenized' which is assumed to be of the format [[],[],...,[]]
        """
        stop_words = set(nltk.corpus.stopwords.words('english'))
        filtered_sentence = []
        if return_type == 'list':
            for sentence in tokenized:
                place_holder = []
                for word in sentence:
                    if word not in stop_words:
                        place_holder.append(word)
                filtered_sentence.append(place_holder)
            return filtered_sentence
        elif return_type == 'tokens':
            for sentence in tokenized:
                for word in sentence:
                    if word not in stop_words:
                        filtered_sentence.append(word)
            return filtered_sentence
