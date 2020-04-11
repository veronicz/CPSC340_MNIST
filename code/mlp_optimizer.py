from sklearn.model_selection import GridSearchCV
from neural_net import NeuralNet


def optimize(X, Y, n_folds, verbose=0):
    parameter_space = {
        'hidden_layer_sizes': [[200], [500]],
        'lammy': [5, 1, 1e-3],
        'alpha': [1e-3, 1e-4],
        'batch_size': [200, 500, 700],
        'epochs': [100, 1000, 10000]
    }
    model = NeuralNet()
    clf = GridSearchCV(model, parameter_space,
                       scoring='neg_log_loss', n_jobs=-1, cv=n_folds)
    clf.fit(X, Y)

    if verbose > 0:
        print('Best parameters found:\n', clf.best_params_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    return clf.best_estimator_, clf.best_params_
