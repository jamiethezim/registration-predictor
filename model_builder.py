import argparse

import pandas
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy

def main(directory):
    df = pandas.read_csv(directory)

    # shape of dataset
    print("Shape:", df.shape)

    # column names
    print("\nFeatures:", df.columns)

    # storing the feature matrix (X) and response vector (y)
    X = df[df.columns[:-1]]
    y = df[df.columns[-1]].to_numpy()

    # printing first 5 rows of feature matrix
    print("\nFeature matrix:\n", X.head())

    # printing first 5 values of response vector
    print("\nResponse vector:\n", y[0:7])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=42
    )

    le = preprocessing.LabelEncoder()
    for column in X_train:
        X_train[column] = le.fit_transform(X_train[column])
    for column in X_test:
        X_test[column] = le.fit_transform(X_test[column])

    rfc = RandomForestClassifier()
    rfc.fit(X_train, y_train)
    print("\nTraining model...\n", "-"*50)
    print("\nModel Accuracy score is: ", rfc.score(X_test, y_test))

    feature_importance = pandas.DataFrame(rfc.feature_importances_, index=X_train.columns).sort_values(by=0, ascending=False)
    print("\nFeature Importance:", feature_importance)

    y_pred = rfc.predict(X_test)
    labels = numpy.unique(y_pred)
    clf_report = classification_report(y_test, y_pred, labels=labels)
    print("\nClassification Report:\n", clf_report)

    conf_mtx = confusion_matrix(y_test,
                                y_pred,
                                labels=labels)

    print("\nConfusion Matrix:")
    print(pandas.crosstab(y_test, y_pred, rownames=[''], colnames=[''], margins=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", "-d", type=str, required=True)
    args = parser.parse_args()
    main(args.directory)
