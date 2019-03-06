import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB, MultinomialNB
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cross_validation import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.feature_selection import RFE, RFECV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from tabulate import tabulate

import csv

def main():
    # classifiers
    linearSVC = SVC(kernel="linear", random_state=1, probability=True)
    mnb = MultinomialNB()
    randomForest = RandomForestClassifier(n_estimators=20)
    classifiers = [mnb, linearSVC, randomForest]
    nFeature = 15
    X_train, y_train, featRanked = preprocessing()


    index = 0
    clfAcurracy = [[], [], []]
    table = [[],[],[],[],[],[]]
    for clf in classifiers:

        for n in np.arange(4,56, 5):
            X_train_n = removeFeatures(X_train, featRanked, n)
            precision, recall, f_score = customCrossValidation(X_train_n, y_train, clf)
            table[index].append(precision)
            table[index+1].append(recall)
            clfAcurracy[int(index/2)].append(f_score)


        index = index + 2
    table = np.around(table, decimals=3)
    table = table * 100
    print("")
    print("\tClassifier   |", "         MNB    ", "|", "         SVM    ", "|", "     RandForest ", "|")
    results = [(x*5, "|", table[0][x-1], table[1][x-1], "|", table[2][x-1],
                table[3][x-1], "|", table[4][x-1],table[5][x-1], "|") for x in range(1, 12)]
    print(tabulate(results, headers=["# of features", "|", "Prec", "Rec", "|", "Prec", "Rec", "|", "Prec", "Rec", "|"]))

    table = table.transpose()
    table = table.reshape(11,6)
    n = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55]
    n = np.array(n).reshape(-1,1)
    table = np.concatenate((n, table), axis=1)

    with open('measures_recorded.csv', 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
		wr.writerow(["n features", "MNB", "", "SVM", "", "RandomForest", ""])
        wr.writerow(["n features", "Prec", "Rec", "Prec", "Rec", "Prec", "Rec"])
        for i in range(0,11):
            wr.writerow(table[i])
    plotFeat_Accu(clfAcurracy)

def preprocessing():
    df = pd.read_csv('features.csv')
    y_data = df['Label']
    x_data = df.drop('Label', axis=1)

    x_data = x_data.drop('WRB', axis=1) #constant 0
    x_data = x_data.drop('NNP', axis=1) #constant 0

    x_data = x_data.drop('TextID', axis=1)
    x_data = x_data.drop('URL', axis=1)
    x_data = x_data.drop("totalWordsCount", axis=1)
    global categories
    categories = x_data.columns.values
    scaler = MinMaxScaler()
    scaler.fit(x_data)
    x_data = scaler.transform(x_data)



    featRanked = rankFeatures(x_data, y_data)




    return x_data, y_data, featRanked

def rankFeatures(X_train, y_train):
    linearSVC = SVC(kernel="linear", random_state=0)
    #linearSVC = RandomForestClassifier(n_estimators=10)
    selector = RFE(estimator=linearSVC, n_features_to_select=1, step=1)
    selector.fit(X_train, y_train)
    ranks = selector.ranking_
    index = 0
    rankIndex = []
    for rank in ranks:
        tup = [index,rank]
        index = index + 1
        rankIndex.append(tup)
    rankIndex = sorted(rankIndex, key=lambda x: x[1])
    featRanked = []
    for feat in rankIndex:
        featRanked.append(feat[0])

    new_features = []
    for i in featRanked:
        new_features.append(categories[i])
    print("Ranked Features: ", new_features)
    return featRanked


def removeFeatures(X_train, featRanked, n):
    featRanked = featRanked[0:n]


    new_features = []
    for i in featRanked:
        new_features.append(categories[i])

    X_train = X_train[:, featRanked]
    return X_train

def customCrossValidation(X, y, classifier, n_folds=4, shuffle=True, random_state=0):
    skf = StratifiedKFold(y, n_folds=n_folds, shuffle=shuffle, random_state=random_state)
    cm = None
    y_predicted_overall = None
    y_test_overall = None
    for train_index, test_index in skf:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        ros = RandomOverSampler(random_state=1)
        X_train, y_train = ros.fit_sample(X_train, y_train)
        X_test, y_test = ros.fit_sample(X_test, y_test)

        classifier.fit(X_train, y_train)
        y_predicted = classifier.predict(X_test)
        # collect the y_predicted per fold
        if y_predicted_overall is None:
            y_predicted_overall = y_predicted
            y_test_overall = y_test
        else:
            y_predicted_overall = np.concatenate([y_predicted_overall, y_predicted])
            y_test_overall = np.concatenate([y_test_overall, y_test])
        cv_cm = metrics.confusion_matrix(y_test, y_predicted)
        # sum the cv per fold
        if cm is None:
            cm = cv_cm
        else:
            cm += cv_cm
    precision, recall, f_score, support = metrics.precision_recall_fscore_support(y_test_overall,
            y_predicted_overall)
    return precision[0], recall[0], f_score[0]


def plotFeat_Accu(clfAccuracy):
    plt.figure(1)
    x_axis = np.arange(4, 56, 5)
    plt.plot(x_axis, clfAccuracy[0], label="MNB")
    plt.plot(x_axis, clfAccuracy[1], label="linearSVM")
    plt.plot(x_axis, clfAccuracy[2], label="RandomForest")
    plt.legend(loc='lower right')
    plt.title("F-Score of MNB, linearSVM, RandomForest after SVM-RFE")
    plt.xlabel("Number of Features")
    plt.ylabel("F-Score")
    plt.show()


main()
