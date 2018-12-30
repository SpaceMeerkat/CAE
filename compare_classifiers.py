from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_classification

import pickle

from matplotlib import pyplot as plt


import numpy as np

def evaluate_clfs(names, classifiers, dataset, visualize=False, plot_train_pts=False):
    if visualize:
        figure = plt.figure(figsize=(32, 3))
        h = .02  # step size in the mesh
    i = 1
    
    # preprocess dataset, split into training and test part
    X_train, y_train, X_test, y_test = dataset
    
    if visualize:
        x_min, x_max = min(X_train[:, 0].min(), X_test[:, 0].min()) - .5, max(X_train[:, 0].max(), X_test[:, 0].max()) + .5
        y_min, y_max = min(X_train[:, 1].min(), X_test[:, 1].min()) - .5, max(X_train[:, 1].max(), X_test[:, 1].max()) + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

        # just plot the dataset first
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(1, len(classifiers) + 1, i)
        ax.set_title("Input data")
        # Plot the training points
        if plot_train_pts:
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                    edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
                edgecolors='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())

    clfs = {}
    scores = {}

    i += 1        
    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        print("Fitting %s" % name)

        if visualize:
            ax = plt.subplot(1, len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        if visualize:
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            if hasattr(clf, "decision_function"):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

            # Plot the training points
            if plot_train_pts:
                ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                        edgecolors='k')
            # Plot the testing points
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                    edgecolors='k', alpha=0.6)

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            ax.set_title(name)
            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                    size=15, horizontalalignment='right')

        clfs[name] = clf
        scores[name] = score
        i += 1

    if visualize:
        plt.tight_layout()
        plt.show()

    return clfs, scores

def preprocess_data(train_path, test_path, two_labels_only=False):
    with open(train_path, "rb") as f:
        train_data = np.asarray(pickle.load(f))
        x_train, y_train = train_data[:, :-1], train_data[:, -1]

    print("%d train samples" % len(x_train))

    with open(test_path, "rb") as f:
        test_data = np.asarray(pickle.load(f))
        x_test, y_test = test_data[:, :-1], test_data[:, -1]

    print("%d test samples" % len(x_test))

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)

    y_train = (y_train > 0.5).astype(np.uint8)
    y_test = (y_test > 0.5).astype(np.uint8)

    if two_labels_only:
        x_train = x_train[:, [0, 1]]
        x_test = x_test[:, [0, 1]]
    return x_train, y_train, x_test, y_test


for two_features in [True, False]: 
    print("Two Features: %s" % two_features)
    clfs = [
        KNeighborsClassifier(),
        SVC(kernel="linear", C=0.025, verbose=True),
        SVC(gamma=2, C=1, verbose=True),
        DecisionTreeClassifier(), 
        ExtraTreesClassifier(n_estimators=300, n_jobs=-1, verbose=True),
        RandomForestClassifier(n_estimators=300, n_jobs=-1, verbose=True),
        GradientBoostingClassifier(n_estimators=300, verbose=True),
        BaggingClassifier(SVC(verbose=False), verbose=True, n_estimators=300, n_jobs=-1),
        AdaBoostClassifier(base_estimator=SVC(verbose=False, probability=True), n_estimators=50),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        MLPClassifier(alpha=1, verbose=True, hidden_layer_sizes=(128, 32, 16), max_iter=500),
        # GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs=-1)
    ]

    names = ["Nearest Neighbors", 
            "Linear SVM", 
            "RBF SVM",
            "Decision Tree", 
            "Extra Trees", 
            "Random Forest", 
            "Gradient Boosting", 
            "Bagging SVC", 
            "AdaBoost SVC", 
            "Native Bayes", 
            "QDA", 
            "Neural Net", 
            "Gaussian Process"]


    OUTPUT_DIR = "/home/schock/Downloads/"
    INPUT_DIR = "/home/schock/Downloads/"
    dataset = preprocess_data(os.path.join(INPUT_DIR, "features_train.pkl"), 
                                os.path.join(INPUT_DIR, "features_test.pkl"), two_features)

    clfs, scores = evaluate_clfs(names, clfs, dataset, visualize=two_features)

    appendix = "_two_features" if two_features else "_all_features"
    with open(os.path.join(OUTPUT_DIR, "clfs%s.pkl" % appendix), "wb") as f:
        pickle.dump(clfs, f)
    with open(os.path.join(OUTPUT_DIR, "scores%s.pkl" % appendix), "wb") as f:
        pickle.dump(scores, f)
        
    print("="* 50 + "\n" + "=" + " "*9 + "Two Features:%s" % two_features + " "*9 + "=" + "\n" + "="*50)
    for key, val in scores.items():
        print("%s : %.3f" % (key, val))    
