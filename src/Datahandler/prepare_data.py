import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler

def data_prep(data):
    X_train, X_test, y_train, y_test = __train_validation_test_split(data)
    X_train, X_test, y_train, y_test = __scale(X_train, X_test, y_train, y_test)
    print("hi")
    return X_train, X_test, y_train, y_test

def __train_validation_test_split(data: pd.DataFrame):
    X_train, X_test, y_train, y_test = train_test_split(data, data, test_size=0.33)
    return X_train, X_test, y_train, y_test


def __scale(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    # The scaler is fit only to the training data, but all datasets are transformed
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    return X_train, X_test, y_train, y_test
