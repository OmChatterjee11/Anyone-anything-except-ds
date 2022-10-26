from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn.grid_search import GridSearchCV
from sklearn import svm
import numpy as np


def get_label(path='./cifar10/train/labels.txt'):
    """ Get cifar10 class label"""
    with open(path,'r') as f:
        names = f.readlines()
    names = [n.strip() for n in names]
    return names



def svm_classifier(x_train, y_train, x_test=None, y_test=None):
    if x_test == None and y_test == None:
        x_train, x_test, y_train, y_test = train_test_split(
                x_train, y_train, test_size=0.2, random_state=6)
        print "Spliting train:{}/test:{} from training data".format(
                len(x_train), len(x_test))
    C_range = 10.0 ** np.arange(-3, 3)
    gamma_range = 10.0 ** np.arange(-3, 3)
    param_grid = dict(gamma=gamma_range.tolist(), C=C_range.tolist())

    # Grid search for C, gamma, 5-fold CV
    print("Tuning hyper-parameters\n")
    clf = GridSearchCV(svm.SVC(), param_grid, cv=5, n_jobs=-2)
    clf.fit(x_train, y_train)
    print("Best parameters set found on development set:\n")
    print(clf.best_estimator_)
    print("\nGrid scores on development set:\n")
    for params, mean_score, scores in clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() * 2, params))
    print("\nDetailed classification report:\n")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.\n")
    y_true, y_pred = y_test, clf.predict(x_test)
    #print(classification_report(y_true, y_pred, target_names=get_label()))
    print(classification_report(y_true, y_pred))
