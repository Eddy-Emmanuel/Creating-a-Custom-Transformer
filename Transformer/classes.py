# importing necessary libraries 
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


# Creating classes
class kfold_validation(BaseEstimator, TransformerMixin):
    # initalizing class 
    def __init__(self, n_folds = 5):
        self.n_folds = n_folds
        # Creating an empty list for classification scores
        self.train_accuracy_score = []
        self.test_accuracy_score = []
        self.train_f1_score = []
        self.test_f1_score = []
        # Creating an empty list for regression score
        self.train_mae = []
        self.test_mae = []
        self.train_rmse = []
        self.test_rmse = []



    # Creating a method that fits the data into the model
    def fit(self, data: pd.DataFrame, target: pd.Series, model, problem_type: str):

        self.model_ = model()

        fold = KFold(n_splits = self.n_folds)

        for (train, test) in fold.split(data, target):
            x_train, x_test = data.iloc[train], data.iloc[test]
            y_train, y_test = target[train], target[test]
            # Training the model
            self.model_.fit(x_train, y_train)

            # Making predictions
            train_pred, test_pred = self.model_.predict(x_train), self.model_.predict(x_test)

            # Storing accuracy score and f1 score in a list if problem type == classification
            if problem_type.lower() == "classification":
                self.train_accuracy_score.append(accuracy_score(y_true = y_train, y_pred = train_pred))
                self.test_accuracy_score.append(accuracy_score(y_true = y_test, y_pred = test_pred))
                self.train_f1_score.append(f1_score(y_true = y_train, y_pred = train_pred))
                self.test_f1_score.append(f1_score(y_true = y_test, y_pred = test_pred))

            # Storing mean absolute error and root mean squared error in a list if problem type == regression
            elif problem_type.lower() == "regression":
                self.train_mae.append(mean_absolute_error(y_true = y_train, y_pred = train_pred))
                self.test_mae.append(mean_absolute_error(y_true = y_test, y_pred = test_pred))
                self.train_rmse.append(np.sqrt(mean_squared_error(y_true = y_train, y_pred = train_pred)))
                self.test_rmse.append(np.sqrt(mean_squared_error(y_true = y_test, y_pred = test_pred)))

            # Raise an error if problem type is neither classification nor regression
            else:
                raise Exception("Problem type can either be a 'regression' or a 'classification' type")

        return self

    # Creating a method that returns the prediction of unseen data
    def predict(self, drop_out_data):
        return self.model_.predict(drop_out_data)
        

    # This method returns the accuracy score of the model on train and test set
    def model_accuracy_score(self):
        return f"Train set accuracy score: {np.mean(self.train_accuracy_score)}\n\nTest set accuracy score: {np.mean(self.test_accuracy_score)}"

    # This method returns the f1 score of the model on train and test set
    def model_f1_score(self):
        return f"Train set f1 score: {np.mean(self.train_f1_score)}\n\nTest set f1 score: {np.mean(self.test_f1_score)}"

    # This method returns the mean absolute error of the model on the train and test set
    def model_mean_absolute_error(self):
        return f"Train mean absolute error: {np.mean(self.train_mae)}\n\nTest mean absolute error: {np.mean(self.test_mae)}"

    # This method returns the root mean square error of the model on the train and test set
    def model_root_mean_squared_error(self):
        return f"Train root mean square error: {np.mean(self.train_rmse)}\n\nTest root mean square error: {np.mean(self.test_rmse)}"



class stratified_kfold_validation(object):

    # initalizing class 
    def __init__(self, n_folds = 5):
        self.n_folds = n_folds
        # Creating an empty list for classification scores
        self.train_accuracy_score = []
        self.test_accuracy_score = []
        self.train_f1_score = []
        self.test_f1_score = []
        # Creating an empty list for regression score
        self.train_mae = []
        self.test_mae = []
        self.train_rmse = []
        self.test_rmse = []


    # Creating a method that fits the data into the model
    def fit(self, data: pd.DataFrame, target: pd.Series, model, problem_type: str):

        self.model_ = model()

        fold = StratifiedKFold(n_splits = self.n_folds)

        for (train, test) in fold.split(data, target):
            x_train, x_test = data.iloc[train], data.iloc[test]
            y_train, y_test = target[train], target[test]
            # Training the model
            self.model_.fit(x_train, y_train)

            # Making predictions
            train_pred, test_pred = self.model_.predict(x_train), self.model_.predict(x_test)

            # Storing accuracy score and f1 score in a list if problem type == classification
            if problem_type.lower() == "classification":
                self.train_accuracy_score.append(accuracy_score(y_true = y_train, y_pred = train_pred))
                self.test_accuracy_score.append(accuracy_score(y_true = y_test, y_pred = test_pred))
                self.train_f1_score.append(f1_score(y_true = y_train, y_pred = train_pred))
                self.test_f1_score.append(f1_score(y_true = y_test, y_pred = test_pred))

            # Storing mean absolute error and root mean squared error in a list if problem type == regression
            elif problem_type.lower() == "regression":
                self.train_mae.append(mean_absolute_error(y_true = y_train, y_pred = train_pred))
                self.test_mae.append(mean_absolute_error(y_true = y_test, y_pred = test_pred))
                self.train_rmse.append(np.sqrt(mean_squared_error(y_true = y_train, y_pred = train_pred)))
                self.test_rmse.append(np.sqrt(mean_squared_error(y_true = y_test, y_pred = test_pred)))

            # Raise an error if problem type is neither classification nor regression
            else:
                raise Exception("Problem type can either be a 'regression' or a 'classification' type")

        return self

    # Creating a method that returns the prediction of unseen data
    def predict(self, drop_out_data):
        return self.model_.predict(drop_out_data)

    # This method returns the accuracy score of the model on train and test set
    def model_accuracy_score(self):
        return f"Train set accuracy score: {np.mean(self.train_accuracy_score)}\n\nTest set accuracy score: {np.mean(self.test_accuracy_score)}"

    # This method returns the f1 score of the model on train and test set
    def model_f1_score(self):
        return f"Train set f1 score: {np.mean(self.train_f1_score)}\n\nTest set f1 score: {np.mean(self.test_f1_score)}"

    # This method returns the mean absolute error of the model on the train and test set
    def model_mean_absolute_error(self):
        return f"Train mean absolute error: {np.mean(self.train_mae)}\n\nTest mean absolute error: {np.mean(self.test_mae)}"

    # This method returns the root mean square error of the model on the train and test set
    def model_root_mean_squared_error(self):
        return f"Train root mean square error: {np.mean(self.train_rmse)}\n\nTest root mean square error: {np.mean(self.test_rmse)}"
    