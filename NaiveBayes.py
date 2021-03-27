import numpy as np
from ProcessDataClass import ProcessData
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from MultinomialNB import MultinomialNB as myMultinomialNB
import pickle


if __name__ == '__main__':
    #Initialize
    FILE_PATH = "data/URL Classification.csv"
    SAVED = True

    process_data = ProcessData()
    x_train, y_train, x_test, y_test = process_data.read_csv(FILE_PATH)

    if SAVED == False:
        t1 = time.time()
        x_train_tfidf = process_data.vectorizer_url(x_train)
        t2 = time.time()
        print(f"time to run process_data.vectorizer_url= {t2 - t1}")
        with open("vectorized_url.pickle", "wb") as handle:
            pickle.dump([process_data, x_train_tfidf], handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open("vectorized_url.pickle", "rb") as handle:
            process_data, x_train_tfidf = pickle.load(handle)

    x_test_tfidf = process_data.vectorizer_url_test_set(x_test)
    print(f"x_train_tfidf shape = {x_train_tfidf.shape}")
    print(f"x_test_tfidf shape = {x_test_tfidf.shape}")

    t1 = time.time()
    clf = myMultinomialNB(alpha=0.001)
    clf.fit(x_train_tfidf, y_train)
    t2 = time.time()
    print(f"time to fit = {t2 - t1}")

    test_str = ["http://usa-aliso.com", "http://oc.app.epa.com"]
    test = process_data.vectorizer_url_test_set(test_str)

    t1 = time.time()
    y_pred = clf.predict(x_test_tfidf)
    accuracy = clf.accuracy_score(y_test, y_pred)
    t2 = time.time()
    print("accuracy of training set = ", accuracy)
    print(f"time to predict = {t2 - t1}")
    print(metrics.classification_report(y_test, y_pred))

    confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(np.array(confusion_matrix), fmt='d', annot=True, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True label")
    plt.show()


