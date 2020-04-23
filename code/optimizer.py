from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from linear_model import MultiClassSVM, SoftmaxClassifier
from neural_net import NeuralNet
from cnn import CNN


def optimizeSVM(X, y, n_folds, verbose=0):
    parameter_space = {
        'lammy': [5, 1, 0.1],
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [1000, 2000],
        'epochs': [500, 1000]
    }

    return optimizeClassifier(MultiClassSVM(), parameter_space, X, y, n_folds, verbose)


def optimizeLR(X, y, n_folds, verbose=0):
    parameter_space = {
        'lammy': [5, 1, 0.1],
        'alphaInit': [1e-3, 1e-4],
        'maxEvals': [100, 500, 1000]
    }

    return optimizeClassifier(SoftmaxClassifier(), parameter_space, X, y, n_folds, verbose)


def optimizeMLP(X, y, n_folds, verbose=0):
    parameter_space = {
        'hidden_layer_sizes': [[200], [500]],
        'lammy': [5, 1, 1e-3],
        'alpha': [1e-3, 1e-4],
        'batch_size': [200, 500, 700],
        'epochs': [100, 1000, 10000]
    }

    return optimizeClassifier(NeuralNet(), parameter_space, X, y, n_folds, verbose)


def optimizeCNN(X, y, n_folds, verbose=0):
    parameter_space = {
        'n1_filters': [5, 8],
        'n2_filters': [5, 8],
        'filter_size': [3, 5, 8],
        'conv_stride': [1, 2],
        'mlp_size': [64, 128],
        'batch_size': [32, 64],
        'epochs': [1, 2],
        'learning_rate': [0.01, 1e-3],
        'beta1': [0.9, 0.95, 0.99]
    }

    return optimizeClassifier(CNN(), parameter_space, X, y, n_folds, verbose, random=True)


def optimizeClassifier(model, parameter_space, X, y, n_folds, verbose, random=False):
    if random:
        clf = RandomizedSearchCV(model, parameter_space, scoring='neg_log_loss',
                                 n_jobs=-1, cv=n_folds, n_iter=50)
    else:
        clf = GridSearchCV(model, parameter_space,
                           scoring='neg_log_loss', n_jobs=-1, cv=n_folds)
    clf.fit(X, y)

    if verbose > 0:
        print('Best parameters found:\n', clf.best_params_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    return clf.best_estimator_, clf.best_params_
