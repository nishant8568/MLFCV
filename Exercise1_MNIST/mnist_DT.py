__author__ = 'Nishant Gupta'

import mnist
import numpy as np
from sklearn.externals.six import StringIO
import pydot
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV

DIR_PATH = "./Dataset"

def score(predictions, labels):
    matches = np.count_nonzero(predictions == labels)
    return float(matches) / labels.shape[0]

def pydotVisualization(clf, filename):
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('./'+ filename + '.png')
    print "tree plotted"

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

    listTestImg = []
    for img in test_data:
        testImg = img.flatten()
        listTestImg.append(list(testImg))

    # Preprocessing
    mean = np.mean(listTrainImg, axis=0)
    print mean
    listTrainImg = listTrainImg - np.tile(mean, [np.asarray(listTrainImg).shape[0], 1])
    listTestImg = listTestImg - np.tile(mean, [np.asarray(listTestImg).shape[0], 1])

    """
    param_grid = {
                   'criterion' : ['entropy'],
                   'max_depth' : [10, 50, 70, 100],
                   'max_features' : [50, 100, 250, 500]
                  }

    classifier = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=5, n_jobs=-1)
    print classifier.fit(listTrainImg,training_labels)
    print classifier.best_params_
    """
    #classifier = tree.DecisionTreeClassifier()
    classifier = tree.DecisionTreeClassifier(criterion='entropy', max_depth=50, max_features=500)
    classifier.fit(listTrainImg,training_labels)
    predictions = classifier.predict(listTestImg)
    print score(predictions, test_labels)

    # pydot Visualization
    #pydotVisualization(classifier, filename="treePlot")
    #pixel importance
    pixelPlot(classifier, filename="treePixels")

if __name__ == "__main__":
    dt_baseline()