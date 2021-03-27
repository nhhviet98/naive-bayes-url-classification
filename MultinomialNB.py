import numpy as np
import math
import scipy.sparse
from tqdm import tqdm


class MultinomialNB():
    def __init__(self, alpha: float = 1.0):
        '''
        Init for MultinomialNB class
        :param alpha: float
        '''
        self.alpha = alpha
        self.prior_prob = None
        self.num_class = None
        self.tfidf_count = None
        self.word_count = None

    def fit(self, x_train, y_train: np.ndarray):
        '''
        Fit trainning set to model
        :param x_train: csr_matrix
            Input of training set
        :param y_train: list
            Output of training set
        '''
        y_train = np.array(y_train)
        self.num_class = np.max(y_train) + 1
        self.prior_prob = self.calculate_prior_prob(y_train)
        self.calculate_tfidf_count(x_train, y_train)

    def calculate_prior_prob(self, y_train: np.ndarray) -> np.ndarray:
        '''
        Calculate prior probability for each class
        :param y_train: np.ndarray
            Output label of traning set
        :return: np.ndarray
            Prior probabilities for each class in training set
        '''
        self.prior_prob = [0]*self.num_class
        for i in range(self.num_class):
            mask = y_train == i
            count_mask = np.count_nonzero(mask)
            self.prior_prob[i] = count_mask
        self.prior_prob /= np.int32(len(y_train))
        return self.prior_prob

    def calculate_tfidf_count(self, x_train, y_train: np.ndarray):
        '''
        Calculate tf-idf count in training phase
        :param x_train: csr_matrix
            Input of training set
        :param y_train: np.ndarray
            Output of training set
        '''
        self.word_count = x_train.shape[1]
        self.tfidf_count = []
        for i in range(self.num_class):
            index_arr = np.where(y_train == i)
            count_sum = x_train[index_arr].sum()
            self.tfidf_count.append((np.array(x_train[index_arr].sum(axis=0)).squeeze() + self.alpha)/(count_sum + self.word_count))

    def predict(self, x_test) -> list:
        '''
        Predict output for test set
        :param x_test: csr_matrix
            Input of test set
        :return: list
            List of predicted classes
        '''
        y_pred = []
        for x in tqdm(x_test):
            prob = [0] * self.num_class
            for i in range(self.num_class):
                probability = 1
                for j in range(len(x.data)):
                    probability *= self.tfidf_count[i][x.indices[j]]**x.data[j]
                prob[i] = math.log(probability*self.prior_prob[i])
            y_pred.append(np.argmax(prob))
        return y_pred

    def accuracy_score(self, y_test: list, y_pred: list) -> float:
        '''
        Calculate accuracy of model
        :param y_test: list
            Labels of test set
        :param y_pred:
            Predicted labels of model
        :return: float
            Accuracy of model
        '''
        return sum(np.equal(y_test, y_pred))/len(y_test)






