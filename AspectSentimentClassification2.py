import pandas as pd
import collections
import nltk.metrics
import sklearn.metrics
from sklearn.datasets import make_classification
from nltk.metrics import precision, recall, f_measure
import re, string
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from xlrd.formula import quotedsheetname
import nltk
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import precision_recall_fscore_support
import functools
from nltk.corpus import stopwords
from sklearn import svm
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy
import numpy as np
import sklearn
from sklearn.model_selection import KFold
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix


corpus = []

listOfSentences = list()
listOfPositiveSentences = list()
listOfNegativeSentences = list()
listOfNeutralSentences = list()

def readCSVFileToTrain():
    with open('data1_train.csv', 'r', encoding='utf8') as csvfile:
        ## use Pandas to read the “Sentiment” column,
        df = pd.read_csv(csvfile)
        print(type(df))
    global listOfSentences
    listOfSentences = list()
    examples = df['example_id']
    text = df[' text']
    text = text.str.lower()
    listOfSentences = list(text)


    # print(listOfSentences)
    # print(len(listOfSentences))
    aspect_term = df[' aspect_term']
    term_location = df[' term_location']
    sentiment = df[' class']
    #classifyPositive(sentiment, text)

    global listOfSentDict
    listOfSentDict = []
    for m in range(len(listOfSentences)):
        item = removeNumbers(text[m])
        item = removeComma(item)
        item = removePunctuationsExpressions(item)
        item = stopwords1(item)
        # item = posTagging(item)
        item = stemming(item)
        case = {'example_id': examples[m], 'text': item, 'aspect_term': aspect_term[m], 'sentiment': sentiment[m]}
        listOfSentDict.append(case)

    listOfPositiveSentences = classifyPositive(listOfSentDict)
    listOfNegativeSentences = classifyNegative(listOfSentDict)
    listOfNeutralSentences = classifyNeutral(listOfSentDict)
    # print(listOfPositiveSentences)
    # print(listOfNegativeSentences)
    # print(listOfNeutralSentences)


    refsets = collections.defaultdict(set)
    testsets = collections.defaultdict(set)

    training = listOfPositiveSentences[:int((.8) * len(listOfPositiveSentences))] + listOfNegativeSentences[:int((.8) * len(listOfNegativeSentences))] + listOfNeutralSentences[:int((.8) * len(listOfNeutralSentences))]
    test = listOfPositiveSentences[int((.8) * len(listOfPositiveSentences)):] + listOfNegativeSentences[int((.8) * len(listOfNegativeSentences)):] + listOfNeutralSentences[:int((.8) * len(listOfNeutralSentences)):]
    textdata = listOfPositiveSentences + listOfNegativeSentences + listOfNeutralSentences

    features = functools.reduce(lambda l1, l2: l1 + l2, text.str.split(' '))
    features_set = list(set(features))

    cv = sklearn.feature_extraction.text.CountVectorizer(vocabulary=features_set)
    count_vectorized = cv.fit_transform(text.values.flatten()).toarray()
    text_train = text_test = count_vectorized
    label_train = label_test = sentiment

    # ----------------------- SVM -----------------------------------
    pipelines_svc = LinearSVC().fit(text_train,label_train)
    pred_svc = cross_val_predict(pipelines_svc, text_test, label_test, cv=10)
    np.mean( label_test == pred_svc)
    print(precision_recall_fscore_support(label_test, pred_svc, labels=[-1,0,1] ))
    print("\n Classification Report \n", classification_report(pred_svc, label_test))
    print("\n Accurary : ", accuracy_score(label_test, pred_svc))

    # ----------------------- Naive Bayes ---------------------------
    pipelines_mnb = MultinomialNB().fit(text_train, label_train)
    pred_mnb = cross_val_predict(pipelines_mnb, text_test, label_test, cv=10)
    np.mean(label_test == pred_mnb)
    print(precision_recall_fscore_support(label_test, pred_mnb, labels=[-1, 0, 1]))
    print("\n Classification Report \n", classification_report(pred_mnb, label_test))
    print("\n Accurary : ", accuracy_score(label_test, pred_mnb))

    # ----------------------- Random Forest -----------------------------------
    pipelines_rfc = RandomForestClassifier().fit(text_train, label_train)
    pred_rfc = cross_val_predict(pipelines_rfc, text_test, label_test, cv=10)
    np.mean(label_test == pred_rfc)
    print(precision_recall_fscore_support(label_test, pred_rfc, labels=[-1, 0, 1]))
    print("\n Classification Report \n", classification_report(pred_rfc, label_test))
    print("\n Accurary : ", accuracy_score(label_test, pred_rfc))

    # ----------------------- Decision Tree -----------------------------------
    pipelines_dtc = RandomForestClassifier().fit(text_train, label_train)
    pred_dtc = cross_val_predict(pipelines_dtc, text_test, label_test, cv=10)
    np.mean(label_test == pred_dtc)
    print(precision_recall_fscore_support(label_test, pred_dtc, labels=[-1, 0, 1]))
    print("\n Classification Report \n", classification_report(pred_dtc, label_test))
    print("\n Accurary : ", accuracy_score(label_test, pred_dtc))


    # SVM classification

    # kf = KFold(n_splits=10)
    # kf.get_n_splits(count_vectorized)
    # class_list = (sentiment.values.flatten())
    # sumf1 = 0
    # accscore = 0
    # f1_scores = []
    # for train_index, test_index in kf.split(count_vectorized):
    #     X_train, X_test = count_vectorized[train_index], count_vectorized[test_index]
    #     y_train, y_test = class_list[train_index], class_list[test_index]
    #     cl=svm.SVC(kernel='linear')
    #
    #     cl.fit(X_train,y_train)
    #     y_pred = cl.predict(X_test)
    #
    #     # f1_scores.append(f1_score(y_test, y_pred, average="macro"))
    #     # print(precision_score(y_test, y_pred, average="macro"))
    #     # print(recall_score(y_test, y_pred, average="macro"))
    #     # print(accuracy_score(y_test, y_pred))
    #     print(precision_recall_fscore_support(y_test, y_pred, average='weighted', labels=[1,0,-1]));


    # for i,j in f1_scores,accscore:
    #     sumf1 += i
    #     accscore += j
    # print("------- F1 Score --------")
    # print(sumf1/10);
    # print("------- Accuracy --------")
    # print(accscore / 10);


        #f1_scores.append(svm_classifier(X_train, X_test, y_train, y_test))



    # classifier = NaiveBayesClassifier.train(training)
    #
    # for i, (feats, label) in enumerate(test):
    #     refsets[label].add(i)
    #     observed = classifier.classify(feats)
    #     testsets[observed].add(i)
    #
    # #print(test)
    # print(nltk.classify.accuracy(classifier, test))

# kf = KFold(n_splits=3)
    # sum = 0
    # print(listOfSentDict)
    # print(training)
    # print(test)
    # # for training, test in kf.split(listOfSentDict):
    #     train_data = np.array(listOfSentDict)[training]
    #     test_data = np.array(listOfSentDict)[test]
    #     classifier = nltk.NaiveBayesClassifier.train(training)
    #     sum += nltk.classify.accuracy(classifier, test_data)
    # average = sum / 3
    # print(average)


# X = np.array(["Science today", "Data science", "Titanic", "Batman"])  # raw text
    # y = np.array([1, 1, 2, 2])  # categories e.g., Science, Movies
    # kf = KFold(n_splits=2)
    # for train_index, test_index in kf.split(X):
    #     x_train, y_train = X[train_index], y[train_index]
    # x_test, y_test = X[test_index], y[test_index]



def classifyPositive(listOfSentDict):
    for m, val in enumerate(listOfSentDict):
        if val['sentiment'] == 1:
            if ([format_sentence(val['text']), 'pos']) not in listOfPositiveSentences:
                listOfPositiveSentences.append([format_sentence(val['text']), 'pos'])

    return listOfPositiveSentences

def classifyNegative(listOfSentDict):
    for m, val in enumerate(listOfSentDict):
        if val['sentiment'] == -1:
            if ([format_sentence(val['text']), 'neg']) not in listOfNegativeSentences:
                listOfNegativeSentences.append([format_sentence(val['text']), 'neg'])
    return listOfNegativeSentences

def classifyNeutral(listOfSentDict):
    for m, val in enumerate(listOfSentDict):
        if val['sentiment'] == 0:
            if ([format_sentence(val['text']), 'neu']) not in listOfNeutralSentences:
                listOfNeutralSentences.append([format_sentence(val['text']), 'neu'])
    return listOfNeutralSentences


def format_sentence(sentence):
    return ({word: True for word in sentence})

def stemming(item):
    from nltk.stem import PorterStemmer

    ps = PorterStemmer()
    item = [[ps.stem(token) for token in sentence.split(" ")] for sentence in item]
    return item
    print(item)


def stopwords1(item):
    listOfStopWords = set()
    listOfStopWords = (
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "you're", "you've", "you'll", "you'd",
            "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "she's", "her", "hers",
            "herself", "it", "it's", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which",
            "who", "whom", "this", "that", "that'll", "these", "those", "am", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
            "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out",
            "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why",
            "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "don't",
            "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren", "aren't", "couldn",
            "couldn't", "didn", "didn't", "doesn", "doesn't", "hadn", "hadn't", "hasn", "hasn't", "haven", "haven't",
            "isn", "isn't", "ma", "mightn", "mightn't", "mustn", "mustn't", "needn", "needn't", "shan", "shan't",
            "shouldn", "shouldn't", "wasn", "wasn't", "weren", "weren't", "won", "won't", "wouldn", "wouldn't")
    listOfSentAfterStopWords = []

    listOfWordTokens = word_tokenize(item)
    sentAfterStopWords = [word for word in listOfWordTokens if not word in listOfStopWords]
    item = sentAfterStopWords
    return item
"""
    for w in listOfWordTokens:
        if w not in listOfStopWords:
           sentAfterStopWords.append(w)
"""



def posTagging(item):
    item = nltk.pos_tag(item)
    return item

def removeNumbers(item):
    res = ''.join([i for i in item if not i.isdigit()])
    item = res
    return item



def removeComma(item):
    item = str(item).replace("[comma]", "")
    return item

def removePunctuationsExpressions(item):
    exclude = set(string.punctuation)
    item = ''.join(ch for ch in item if ch not in exclude)
    return item


def main():
    print("Hello, Loading the Classifier")
    readCSVFileToTrain()


if __name__ == '__main__':
    main()