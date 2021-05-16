import pickle
import numpy as np
import pandas as pd
# from joblib import dump

from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import ensemble
import sklearn.neural_network as nn

# disable warnings
import warnings
warnings.filterwarnings('ignore')


def load_data(train_frac=0.7, seed=1):
    """Shuffle the data and randomly split into train and test sets;
       separate the class labels from the features.

    :param path: path where the csv file is stored
    :param train_frac: The decimal fraction of data that should be training data
    :param seed: Random seed for shuffling and reproducibility, default = 1
    :return: Two tuples (in order): (train_features, train_labels), (test_features, test_labels)
    """
    # shuffle and split the data
    # https://numpy.org/doc/stable/reference/random/generated/numpy.random.shuffle.html
    np.random.seed(seed)  # from the function input, seed = 1, this is for the output consistency

    df = pd.read_csv('heart_failure_clinical_records_dataset.csv')
    df_np = df.to_numpy()
    np.random.shuffle(df_np)  # perform shuffle

    # define training data size
    train_size = int(train_frac * df_np.shape[0])
    x_train = df_np[:train_size, :-1]
    y_train = df_np[:train_size, -1]  # only keep the last label column

    # define testing set
    x_test = df_np[train_size:, :-1]
    y_test = df_np[train_size:, -1]

    # convert array to df type
    x_train = pd.DataFrame(x_train)
    y_train = pd.DataFrame(y_train)
    x_test = pd.DataFrame(x_test)
    y_test = pd.DataFrame(y_test)

    # add column names
    columns_name = df.columns[:-1]
    x_train.columns = columns_name
    x_test.columns = columns_name

    return (x_train, y_train), (x_test, y_test)


(X_train, y_train), (X_test, y_test) = load_data()

# Train LogisticRegression
lr_clf = LogisticRegression(random_state=42)
lr_clf.fit(X_train, y_train)
# dump(lr_clf, './models/logistic_regression.joblib')
pickle_out = open("logistic_regression.pkl", "wb")
pickle.dump(lr_clf, pickle_out)
pickle_out.close()

# Train Random Forest
rf_clf = ensemble.RandomForestClassifier(random_state=42)
rf_clf.fit(X_train, y_train)
# dump(rf_clf, './models/random_forest.joblib')
pickle_out = open("random_forest.pkl", "wb")
pickle.dump(rf_clf, pickle_out)
pickle_out.close()

# Train Decision Tree
dt_clf = tree.DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_train)
# dump(dt_clf, './models/decision_tree.joblib')
pickle_out = open("decision_tree.pkl", "wb")
pickle.dump(dt_clf, pickle_out)
pickle_out.close()

# Train Extra-trees Classifier
ext_clf = ensemble.ExtraTreesClassifier(random_state=42)
ext_clf.fit(X_train, y_train)
# dump(ext_clf, './models/extra_trees.joblib')
pickle_out = open("extra_trees.pkl", "wb")
pickle.dump(ext_clf, pickle_out)
pickle_out.close()

# Train Neural Networks MLPClassifier
mlp_clf = nn.MLPClassifier(random_state=42)
mlp_clf.fit(X_train, y_train)
# dump(mlp_clf, './models/neural_networks.joblib')
pickle_out = open("neural_networks.pkl", "wb")
pickle.dump(mlp_clf, pickle_out)
pickle_out.close()

# Train ensemble Voting-Classifier
voting_classifier = ensemble.VotingClassifier([
                                            ('lr_clf', LogisticRegression(random_state=42)),
                                            ('rf_clf', ensemble.RandomForestClassifier(random_state=42)),
                                            ('dt_clf', tree.DecisionTreeClassifier(random_state=42)),
                                            ('ext_clf', ensemble.ExtraTreesClassifier(random_state=42))
                                            ], voting='hard')
voting_classifier.fit(X_train, y_train)
# dump(voting_classifier, './models/voting_classifier.joblib')
pickle_out = open("voting_classifier.pkl", "wb")
pickle.dump(voting_classifier, pickle_out)
pickle_out.close()
