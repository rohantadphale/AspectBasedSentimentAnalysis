import pandas as pd
import collections
import nltk.metrics
import sklearn.metrics
from sklearn.datasets import make_classification
from nltk.metrics import precision, recall, f_measure
from nltk.stem import WordNetLemmatizer
from sklearn.naive_bayes import MultinomialNB
from nltk.classify import NaiveBayesClassifier
import re
import re, string
from xlrd.formula import quotedsheetname
import nltk
from sklearn.svm import LinearSVC
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, \
    confusion_matrix
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier


sumRFAccuracy=0
sumRFPrecision=0
sumRFRecall_score=0
sumRFF1_score=0

my_class_weight = {1: 1, 0: 2, -1: 1}

nltk.download('wordnet')


def readCSVFileToTrain():
    with open('data1_train.csv', 'r', encoding='utf8') as csvfile:
        # use Pandas to read the “Sentiment” column,
        df = pd.read_csv(csvfile)

    with open('Data-1_test.csv', 'r', encoding='utf8') as csvfile:
        # use Pandas to read the “Sentiment” column,
        df0 = pd.read_csv(csvfile)

    global listOfSentences
    listOfSentences = list()
    temp = df['example_id']
    temp = temp.str.lower()
    text = df[' text']
    text = text.str.lower()
    text11 = text
    listOfSentences = list(text)
    listOfText = pd.Series()
    aspect_term = df[' aspect_term']
    term_location = df[' term_location']
    sentiment = df[' class']



    global listOfSentences0
    listOfSentences0 = list()
    temp0 = df0['example_id']
    temp0 = temp0.str.lower()
    texta = df0[' text']
    texta= texta.str.lower()
    listOfSentences0 = list(texta)
    listOfText0 = pd.Series()
    aspect_term0 = df0[' aspect_term']
    term_location0 = df0[' term_location']


    global listOfSentDict
    listOfSentDict = []



    for m in range(len(listOfSentences)):
        item = removeNumbers(text[m])
        item = item.lower()
        item = removeComma(item)
        item = removePunctuationsExpressions(item)
        #item = posTagging(item)
        item = stopwords1(item)
        item = convertBagOfWordsTostring(item)
        item = aspectBasedSent(item, aspect_term[m].lower())
        item=list(item)
        text[m] = convertBagOfWordsTostring(item)
       # temp[m] = setValueToTemp()
        # item = stemming(item)
        case = {'text': item, 'aspect_term': aspect_term[m].lower(), 'sentiment': sentiment[m]}
        listOfSentDict.append(case)



    for m in range(len(listOfSentences0)):
        item = removeNumbers(texta[m])
        item = item.lower()
        item = removeComma(item)
        item = removePunctuationsExpressions(item)
        #item = posTagging(item)
        item = stopwords1(item)
        item = convertBagOfWordsTostring(item)
        item = aspectBasedSent(item, aspect_term[m].lower())
        item=list(item)
        texta[m] = convertBagOfWordsTostring(item)
       # temp[m] = setValueToTemp()
        # item = stemming(item)



    # print(type(temp))
    # print(type(text))
    features = functools.reduce(lambda l1, l2: l1 + l2, text.str.split(' '))
    features_set = list(set(features))
    #features_set= stopwords1(features_set)
    features_set = np.unique(features_set)

    tfidf_vectorizer = TfidfVectorizer(
        min_df=1,  # min count for relevant vocabulary
        max_features=4000,  # maximum number of features
        strip_accents='unicode',  # replace all accented unicode char
        # by their corresponding  ASCII char
        analyzer='word',  # features made of words
        #token_pattern=r'\w{1,}',  # tokenize only words of 4+ chars
        ngram_range=(1, 1),  # features made of a single tokens
        use_idf=True,  # enable inverse-document-frequency reweighting
        smooth_idf=True,  # prevents zero division for unseen words
        sublinear_tf=True,
        vocabulary=features_set)
    ###tfidf_df = tfidf_vectorizer.fit_transform(text11.values.flatten()).toarray()
    newtfidf_df  = tfidf_vectorizer.fit_transform(text11.values.flatten()).toarray()
    tfidfvocab = tfidf_vectorizer.get_feature_names()
    testtfidf = tfidf_vectorizer.transform(texta.values.flatten()).toarray()
    text = list(text)
    for text1 in text:
        texttokens = word_tokenize(text1)
        for token in texttokens:
            newtfidf_df[text.index(text1)][tfidfvocab.index(token)]*=1.1

    print("----------------------------------------------------------------------------")

    count = 0

    target_names = sentiment

    global sumRFAccuracy
    global sumRFPrecision
    global sumRFRecall_score
    global sumRFF1_score

    X_train = newtfidf_df
    X_test = testtfidf
    y_train = y_test = sentiment

    print("------------------------------------S V C----------------------------------------")

    rf = LinearSVC(class_weight="balanced")
    rf.fit(X_train, y_train);
    y_pred_Random_Forest = rf.predict(X_test)

    x = 1
    with open('temp1.txt', 'w') as file:
        for i in y_pred_Random_Forest:
            file.write(str(x) + ';;' + str(i) + '\n')
            x += 1
        file.close()


def setValueToTemp():
    stringtemp = "3 2 1 0 1 2 3"
    return stringtemp
def convertBagOfWordsTostring(item):
    stringOfArray = ""

    for i in item:
        stringOfArray = stringOfArray.strip() +" "+i
    return stringOfArray
def posTagging(item):
    item = re.findall(r'\w+', item)
    item = nltk.pos_tag(item)
    item1 = list()
    lemmatizer = WordNetLemmatizer()
    for index, g in enumerate(item):
        if g[1] == 'NN' or g[1] == 'NNS' or g[1] == 'NNP' or g[1] == 'NNPS' or g[1] == 'JJ' or g[1] == 'JJR' or g[1] == 'JJS' or g[1] == 'IN':
            item2 = lemmatizer.lemmatize(g[0])
            item1.append(item2)

    return item1
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
        "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "should", "should've", "now", "d", "ll", "m", "o", "re", "ve", "y", "ma")
    prepositions = ("with" , "at", "from", "into", "during", "including", "until", "against", "among", "throughout", "despite", "towards", "upon", "concerning", "of", "to", "in", "for", "on", "by", "about", "like", "through", "over", "before", "between", "after", "since", "without", "under", "within", "along", "following", "across", "behind", "beyond", "plus", "except", "but", "up", "out", "around", "down", "off", "above", "near", "'s")
    finList = set(listOfStopWords) - set(prepositions)
    # sentAfterStopWords = [word for word in item if not word in finList]
    sentAfterStopWords = word_tokenize(item)
    newBagOfWords = list()
    for i in range(len(sentAfterStopWords)):
        if sentAfterStopWords[i] not in finList:
            newBagOfWords.append(sentAfterStopWords[i])
    item = newBagOfWords
    return item


def aspectBasedSent(item, aspect_term):

    wordsBeforeAspect = list()
    #listOfWordTokens = word_tokenize(item)
    if aspect_term in item:
        aspect = word_tokenize(aspect_term)
        listOfWordTokens = word_tokenize(item)
        if aspect[0] in listOfWordTokens:
            ind = listOfWordTokens.index(aspect[0])


            if ind - 5 < 0:
                for j in range(0, ind):
                    wordsBeforeAspect.append(listOfWordTokens[j])
            else:
                for j in range(ind-5, ind):
                    wordsBeforeAspect.append(listOfWordTokens[j])
            if len(listOfWordTokens) < ind+5:
                for i in range(ind, len(listOfWordTokens)):
                    wordsBeforeAspect.append(listOfWordTokens[i])
            else:
                for i in range(ind, ind+5):
                    wordsBeforeAspect.append(listOfWordTokens[i])
        return wordsBeforeAspect
    else:
        return word_tokenize(item)



"""
def bigram_word_feats(words, score_fn=BigramAssocMeasures.chi_sq, n=200):
    bigram_finder = BigramCollocationFinder.from_words(words)
    bigrams = bigram_finder.nbest(score_fn, n)
    return dict([(ngram, True) for ngram in itertools.chain(words, bigrams)])
"""
def main():
    print("Hello, Loading the Classifier")
    np.set_printoptions(threshold=np.inf)
    readCSVFileToTrain()


if __name__ == '__main__':
    main()