__author__ = 'Nishant Gupta'

import mnist
import numpy as np
from sklearn.externals.six import StringIO
import pydot
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV

DIR_PATH = "./Dataset"

def score(predictions, labels):
    matches = np.count_nonzero(predictions == labels)
    return float(matches) / labels.shape[0]

def pixelPlot(clf, filename):
    importance = clf.feature_importances_
    importance = importance.reshape(28, 28)
    plt.matshow(importance, cmap=plt.cm.hot)
    plt.title('Pixel importance')
    plt.savefig('./' + filename + '.png' )
    print "pixel importance plotted"

def dt_baseline():
    training_data, training_labels = mnist.load_mnist('training', path=DIR_PATH, selection=slice(0, 60000))
    test_data, test_labels = mnist.load_mnist('testing', path=DIR_PATH, selection=slice(0, 10000))
    #mnist.show(training_data[0])

    listTrainImg = []
    for img in training_data:
        trainImg = img.flatten()
        listTrainImg.append(list(trainImg))

    print len(listTrainImg[0])
    print "train labels"
    print len(training_labels)

    listTestImg = []
    for img in test_data:
        testImg = img.flatten()
        listTestImg.append(list(testImg))

    # Preprocessing
    mean = np.mean(listTrainImg, axis=0)
    print mean
    listTrainImg = listTrainImg - np.tile(mean, [np.asarray(listTrainImg).shape[0], 1])
    listTestImg = listTestImg - np.tile(mean, [np.asarray(listTestImg).shape[0], 1])

    print "Training begins..."

    """
    param_grid = {
                   'criterion' : ['entropy'],
                   'max_depth' : [10, 50, 70, 100],
                   'max_features' : [50, 100, 250, 500],
                   'n_estimators' : [10, 25, 50, 100]
                  }

    classifier = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, n_jobs=-1)
    classifier.fit(listTrainImg,training_labels)
    print classifier.best_params_
    """

    #classifier = RandomForestClassifier()
    classifier = RandomForestClassifier(max_features=50, n_estimators=50, criterion='entropy', max_depth=50)
    classifier.fit(listTrainImg,training_labels)
    predictions = classifier.predict(listTestImg)
    print score(predictions, test_labels)

    # pixel importance
    pixelPlot(classifier, filename="forestPixels")

if __name__ == "__main__":
    dt_baseline()
