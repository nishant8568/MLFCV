__author__ = 'Nishant Gupta'

# Libraries
import mnist
import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

DIR_PATH = "./Dataset"

def score(predictions, labels):
    matches = np.count_nonzero(predictions == labels)
    return float(matches) / labels.shape[0]

def svm_baseline():
    training_data, training_labels = mnist.load_mnist('training', path=DIR_PATH, selection=slice(0, 60000))
    test_data, test_labels = mnist.load_mnist('testing', path=DIR_PATH, selection=slice(0, 10000))

    listTrainImg = []
    for img in training_data:
        trainImg = img.flatten()
        listTrainImg.append(list(trainImg))

    listTestImg = []
    for img in test_data:
        testImg = img.flatten()
        listTestImg.append(list(testImg))

    # Preprocessing
    print "preprocessing"
    mean = np.mean(listTrainImg, axis=0)
    listTrainImg = listTrainImg - np.tile(mean, [np.asarray(listTrainImg).shape[0], 1])
    listTestImg = listTestImg - np.tile(mean, [np.asarray(listTestImg).shape[0], 1])

    # train
    # Default SVM
    # classifier = svm.SVC()

    # Parameter tuning using GridSearchCV
    param_grid = [
    {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
    {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
    {'C': [1, 10, 100, 1000], 'degree': [2, 3, 5], 'kernel': ['poly']},
    ]
    #classifier = GridSearchCV(svm.SVC(cache_size=2000), param_grid, cv=5, n_jobs=-1)

    # Polynomial Kernel
    classifier = svm.SVC(kernel='poly', degree=2, C=1000)

    classifier.fit(listTrainImg, training_labels)
    #print classifier.best_params_
    predictions = classifier.predict(listTestImg)
    print predictions
    print score(predictions, test_labels)

if __name__ == "__main__":
    svm_baseline()