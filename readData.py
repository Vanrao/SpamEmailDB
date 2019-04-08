from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, precision_recall_fscore_support
import numpy as np
import pandas as pd
import warnings
import SpamConstants as sc


class ClassifySpam:
    """ Classify Email Spam or Not Spam from UCI Processed Dataset"""
    columns_header_list = []
    warnings.filterwarnings(sc.WARNING_IGNORE)
    res_dataframe = {sc.TRUE_NEGATIVE: [], sc.FALSE_POSITIVE: [],
                     sc.FALSE_NEGATIVE: [], sc.TRUE_POSITIVE: [],
                     sc.FALSE_POSITIVE_RATE: [], sc.FALSE_NEGATIVE_RATE: [], sc.OVERALL_ERROR_RATE: [],
                     sc.MEAN_ABSOLUTE_ERROR: [], sc.MEAN_SQUARED_ERROR: [],
                     sc.PRECISION: [], sc.RECALL: [], sc.F1_SCORE: []}

    stats_dataframe = {sc.AVG_FALSE_NEG_RATE: [], sc.AVG_FALSE_POS_RATE: [],
                       sc.AVG_OVERALL_ERROR_RATE: [],
                       sc.AVG_PRECISION: [], sc.AVG_RECALL: [], sc.AVG_F1_SCORE: []}

    def createHeaders(self):
        """ Create headers for the UCI dataset file"""
        for i in range(1, sc.WORD_FREQ_COUNT):
            self.columns_header_list.append(sc.WORD_FREQUENCY+str(i))
        for i in range(1,sc.CHAR_FREQ_COUNT):
            self.columns_header_list.append(sc.CHAR_FREQUENCY+str(i))
        self.columns_header_list.append(sc.CAP_RUN_LENGTH_AVG)
        self.columns_header_list.append(sc.CAP_RUN_LENGTH_LONGEST)
        self.columns_header_list.append(sc.CAP_RUN_LENGTH_TOTAL)
        self.columns_header_list.append(sc.CLASS_LABEL)

    def readInputData(self, filename):
        """ Read the data into Pandas Dataframe"""
        pd_dataframe = pd.read_csv(filename, delimiter=sc.DELIMITER, names = self.columns_header_list)
        return pd_dataframe

    def createNaiveBayesModel(self, pd_dataframe):
        """ Multinomial Naive Bayes model creation"""
        clf = MultinomialNB()
        train_set = np.array(pd_dataframe)
        test_set = np.array(pd_dataframe[sc.CLASS_LABEL])
        return clf, train_set, test_set, pd_dataframe

    def buildKFoldCV(self, clf, train_set, test_set, pd_dataframe):
        """ KFold cross validation with confusion matrix"""
        kf = KFold(n_splits=10)
        for train_index, test_index in kf.split(pd_dataframe):
            X_train, X_test = train_set[train_index], train_set[test_index]
            Y_train, Y_test = test_set[train_index], test_set[test_index]

            clf.fit(X_train, Y_train)

            Y_Predicted = clf.predict(X_test)

            True_Neg, False_Pos, False_Neg, True_Pos = confusion_matrix(Y_test, Y_Predicted).ravel()
            self.res_dataframe[sc.TRUE_NEGATIVE].append(True_Neg)
            self.res_dataframe[sc.FALSE_POSITIVE].append(False_Pos)
            self.res_dataframe[sc.FALSE_NEGATIVE].append(False_Neg)
            self.res_dataframe[sc.TRUE_POSITIVE].append(True_Pos)

            mean_abs_error = mean_absolute_error(Y_test, Y_Predicted)
            self.res_dataframe[sc.MEAN_ABSOLUTE_ERROR].append(mean_abs_error)

            mean_sq_error = mean_squared_error(Y_test, Y_Predicted)
            self.res_dataframe[sc.MEAN_SQUARED_ERROR].append(mean_sq_error)

            precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, Y_Predicted, average='macro')
            self.res_dataframe[sc.PRECISION].append(precision)
            self.res_dataframe[sc.RECALL].append(recall)
            self.res_dataframe[sc.F1_SCORE].append(f1_score)

            # FPR = FP/FP+TN
            if False_Pos == 0:
                false_pos_rate = 0
            else:
                false_pos_rate = False_Pos/float(False_Pos+True_Neg)
            self.res_dataframe[sc.FALSE_POSITIVE_RATE].append(false_pos_rate)

            # FNR = FN/FN+TP
            if False_Neg == 0:
                false_neg_rate = 0
            else:
                false_neg_rate = False_Neg/float(False_Neg+True_Pos)
            self.res_dataframe[sc.FALSE_NEGATIVE_RATE].append(false_neg_rate)

            # Overall Misclassification = (FP+FN) / (TP+TN+FP+FN)
            overall_error_rate = (False_Pos+False_Neg)/(True_Neg + False_Pos + False_Neg + True_Pos)
            self.res_dataframe[sc.OVERALL_ERROR_RATE].append(overall_error_rate)
        print(self.res_dataframe)

    def evaluateModel(self):
        """ Average of all error rates"""
        avg_fnr = sum(self.res_dataframe[sc.FALSE_NEGATIVE_RATE])/float(len(self.res_dataframe[sc.FALSE_NEGATIVE_RATE]))
        self.stats_dataframe[sc.AVG_FALSE_NEG_RATE]=avg_fnr
        avg_fpr = sum(self.res_dataframe[sc.FALSE_POSITIVE_RATE])/len(self.res_dataframe[sc.FALSE_POSITIVE_RATE])
        self.stats_dataframe[sc.AVG_FALSE_POS_RATE]=avg_fpr
        avg_overall_rate = sum(self.res_dataframe[sc.OVERALL_ERROR_RATE])/len(self.res_dataframe[sc.OVERALL_ERROR_RATE])
        self.stats_dataframe[sc.AVG_OVERALL_ERROR_RATE]=avg_overall_rate
        avg_precision = sum(self.res_dataframe[sc.PRECISION])/len(self.res_dataframe[sc.PRECISION])
        self.stats_dataframe[sc.AVG_PRECISION]=avg_precision
        avg_recall = sum(self.res_dataframe[sc.RECALL])/len(self.res_dataframe[sc.RECALL])
        self.stats_dataframe[sc.AVG_RECALL]=avg_recall
        avg_f1_score = sum(self.res_dataframe[sc.F1_SCORE])/len(self.res_dataframe[sc.F1_SCORE])
        self.stats_dataframe[sc.AVG_F1_SCORE]=avg_f1_score

        print(self.stats_dataframe)


if __name__ == '__main__':
    classifySpamObj = ClassifySpam()
    classifySpamObj.createHeaders()
    dataset_filename = 'spambase/spambase.data'
    pd_dataframe = classifySpamObj.readInputData(dataset_filename)
    model, train_set, test_set, dataframe = classifySpamObj.createNaiveBayesModel(pd_dataframe)
    classifySpamObj.buildKFoldCV(model, train_set, test_set, dataframe)
    classifySpamObj.evaluateModel()
