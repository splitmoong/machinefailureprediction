from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import pandas as pd


def random_forest(dataset: pd.DataFrame):
    X = dataset.drop(columns=['UDI', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])
    y = dataset['Machine failure']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=14)
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    rf_clf = RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100)
    rf_clf.fit(X_train, y_train)

    y_pred = rf_clf.predict(X_test)

    #evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    metrics = {'accuracy': accuracy, 'f1': f1, 'precision': precision, 'recall': recall}
    return  rf_clf, metrics
