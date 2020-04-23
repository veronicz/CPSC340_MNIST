from sklearn.model_selection import RandomizedSearchCV
from cnn import CNN


def optimize(X, Y, n_folds, verbose=0):
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

    model = CNN()
    clf = RandomizedSearchCV(model, parameter_space, scoring='neg_log_loss',
                             n_jobs=-1, cv=n_folds, n_iter=50)
    clf.fit(X, Y)

    if verbose > 0:
        print('Best parameters found:\n', clf.best_params_)
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.5f (+/-%0.05f) for %r" % (mean, std * 2, params))

    return clf.best_estimator_, clf.best_params_
