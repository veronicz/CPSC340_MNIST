from sklearn.model_selection import GridSearchCV
from linear_model import MultiClassSVM, SoftmaxClassifier


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


def optimizeClassifier(model, parameter_space, X, y, n_folds, verbose):
    clf = GridSearchCV(model, parameter_space, scoring='neg_log_loss',
                       n_jobs=-1, cv=n_folds)
    clf.fit(X, y)

    if verbose > 0:
        print('Best parameters found:\n', clf.best_params_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    return clf.best_estimator_, clf.best_params_
