import pandas as pd


class ReadData:
    """pass in filepath as argument to class variable when instantiating"""

    def __init__(self, filepath):
        self.filePath = filepath

    def read_json(self):
        pd.set_option('display.max_columns', None)
        df = pd.read_json(self.filePath, lines='True')
        return df

    def read_excel(self):
        pd.set_option('display.max_columns', None)
        df = pd.read_csv(self.filePath)
        return df
