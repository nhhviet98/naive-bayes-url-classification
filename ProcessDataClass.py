import pandas as pd
from nltk.tokenize import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import wordninja
import numpy as np


class ProcessData:
    def __init__(self):
        '''
        Create stop words for URL
        '''
        self.stop_words = stopwords.words('english')
        url_stop_words = ['http', 'www', 'net', 'com', 'org',
                          'index', 'htm', 'html']
        self.stop_words += url_stop_words
        self.vectorizer = TfidfVectorizer()

    def read_csv(self, file_path: str) -> tuple:
        '''
        :param file_path: str
            file path of data file
        :return: tuple of list
            x_train, y_train, x_test, y_test
        '''
        df = pd.read_csv(file_path, names=['idx', 'URL', 'Category'])
        np.random.seed(3010)
        train, test = train_test_split(df, test_size=0.05)
        self.x_train = train['URL']
        self.y_train = train['Category'].astype('category').cat.codes.to_list()[:]
        self.x_train = self.x_train.to_list()[:]
        self.x_test = test['URL']
        self.y_test = test['Category'].astype('category').cat.codes.to_list()[:]
        self.x_test = self.x_test.to_list()[:]
        return self.x_train, self.y_train, self.x_test, self.y_test

    def tokenzier_url(self, url_list: list) -> list:
        '''
        Tokenize url by split special character
        :param url_list: list
            List of URL in dataset
        :return: list
            List of words after tokenize
        '''
        #tokenizer = RegexpTokenizer(r'\w+')
        tokenizer = RegexpTokenizer(r'[a-zA-Z0-9]+')
        x_train_tokenized = []
        for x in url_list:
            try:
                x_tokenized = tokenizer.tokenize(x)
                x_tokenized = [w for w in x_tokenized if not w in self.stop_words]
                for word in x_tokenized:
                    word_split = wordninja.split(word)
                    x_tokenized = [w for w in word_split if (1 < len(w) < len(word))] + x_tokenized
                #print(x_tokenized)
                x_tokenized = " ".join(map(str, x_tokenized))
                x_train_tokenized.append(x_tokenized)
            except:
                x_train_tokenized.append('none')
                print(f"error at: {x}")
        return x_train_tokenized

    def vectorizer_url(self, x: list):
        '''
        Calculate tf-idf for training set
        :param x: list
            List of input URL
        :return: csr_matrix
            tf-idf of training set
        '''
        x_tokenized = self.tokenzier_url(x)
        x_tfidf = self.vectorizer.fit_transform(x_tokenized)
        print("Number of word = ", len(self.vectorizer.get_feature_names()))
        return x_tfidf

    def vectorizer_url_test_set(self, x_test: list):
        '''
        calculate tf-idf for test set
        :param x_test: list
            List of input URL in test set
        :return: csr_matrix
            tf-idf of test set
        '''
        x_test_tokenized = self.tokenzier_url(x_test)
        x_test_tfidf = self.vectorizer.transform(x_test_tokenized)
        return x_test_tfidf
