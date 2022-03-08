import os
import sys
import warnings

import numpy as np
import pandas as pd
from pyod.models.cof import COF
from pyod.models.ocsvm import OCSVM
from pyod.models.knn import KNN
from pyod.models.mcd import MCD
from pyod.models.pca import PCA as PCA_pyod
from pyod.models.loda import LODA
from pyod.models.copod import COPOD
from pyod.models.ecod import ECOD
from pyod.models.so_gaal import SO_GAAL

from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore")

stdout = sys.stdout

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')


# Restore
def enablePrint():
    sys.stdout = stdout


class OutlierGridSearcher(object):
    def __init__(self, train_func, higher_is_better=True, scoring_weights=None, algs_to_skip=None, verbose=False):
        """
        Init function

        :param train_func: The training function, should be a function that gets X,y and returns array of scores that
                will be optimized, the calculated mean of the scores would be our score to optimized.

        :param higher_is_better: If we want to find the max or the min of the scores.

        :param scoring_weights: Weights of the scores, if one score from the returned list from the training func is
                more important we can give weights to be used when calculating the mean.

        :param algs_to_skip: List og algorithms we don't want to test.

        :param verbose: If to print the outliers we dropped.

        """
        self._train_func = train_func
        self._higher_is_better = higher_is_better
        self._scoring_weights = scoring_weights
        self._algs = [COF(), OCSVM(), KNN(), MCD(), PCA_pyod(), LODA(), COPOD(), ECOD(), SO_GAAL()]
        self._algs_names = ['COF', 'OCSVM', 'KNN', 'MCD', 'PCA', 'LODA', 'COPOD', 'ECOD', 'SO_GAAL']
        self._verbose = verbose
        if algs_to_skip is not None:
            # deleting the algorithms we don't want to test
            del self._algs[algs_to_skip]
            del self._algs_names[algs_to_skip]

    def _get_outliers_scores(self, alg, top_num, X):
        # Creating a score column
        X["score"] = (alg.decision_scores_ / sum(alg.decision_scores_)).tolist()
        drop_rows = X.sort_values("score", ascending=False).head(top_num)
        X.drop(["score"], axis=1)
        # returning a list of indexes to drop from the df
        return drop_rows.index.values.tolist()

    def _remove_outliers(self, Outlier_indexes, X, y=None, verbose=False, visualize=False):

        pca = PCA(n_components=2)
        principalComponents = pca.fit_transform(X)
        if verbose:
            # prints the lines we remove from the df
            print("droping the following rows:")
            print(X.iloc[Outlier_indexes])
        principalDf = pd.DataFrame(data=principalComponents
                                   , columns=['principal component 1', 'principal component 2'])
        if visualize:
            plt.title("Data visualization")
            plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'], label='sample')
            red_points = principalDf.iloc[Outlier_indexes]
            plt.scatter(red_points['principal component 1'], red_points['principal component 2'],
                        c='red', label='outlier')
            plt.legend()
            # shows a PCA plot of the indexes we remove
            plt.show()

        if y is not None:
            y_dropped = y.drop(labels=Outlier_indexes, axis=0)
        else:
            y_dropped = y
        X_dropped = X.drop(labels=Outlier_indexes, axis=0)
        return X_dropped, y_dropped

    def fit(self, X, y=None, top_nums_to_remove=[0, 5, 10, 20, 30]):
        scores = []
        score_zero = -1
        score_zero_recorded = False

        for i, _alg in enumerate(self._algs):
            enablePrint()
            print("testing " + self._algs_names[i])
            blockPrint()

            if y is not None:
                _alg.fit(X, y)
            else:
                _alg.fit(X)

            scores_nums = {}

            for top_num in top_nums_to_remove:
                # getting outliers scores
                Outlier_indexes = self._get_outliers_scores(_alg, top_num, X)

                # removing suspected outliers
                X_dropped, y_dropped = self._remove_outliers(Outlier_indexes, X, y)

                # running the training func and getting the scores
                score_nums = self._train_func(X_dropped, y_dropped)

                # appending mean to the scores
                if self._scoring_weights is not None:
                    scores_to_write = np.average(score_nums, weights=self._scoring_weights)
                else:
                    scores_to_write = np.average(score_nums)

                if top_num == 0:
                    if score_zero_recorded:
                        scores_to_write = score_zero
                    else:
                        score_zero_recorded = True
                        score_zero = scores_to_write

                scores_nums[top_num] = scores_to_write

            scores.append(scores_nums)

        models_score = pd.DataFrame(index=self._algs_names, data=scores)

        enablePrint()

        if self._higher_is_better:
            best_score = models_score.stack().index[np.argmax(models_score.values)]
            worst_score = models_score.stack().index[np.argmin(models_score.values)]
        else:
            best_score = models_score.stack().index[np.argmin(models_score.values)]
            worst_score = models_score.stack().index[np.argmax(models_score.values)]

        print("\n\n### OUTPUT ###\nThe best score was achieved using " + best_score[0] + " when filtering " + str(
            best_score[1]) + " outliers")
        print("Worst score was " + str(worst_score))
        print(models_score)
        # getting outliers scores
        Outlier_indexes = self._get_outliers_scores(self._algs[self._algs_names.index(best_score[0])], best_score[1], X)

        # removing suspected outliers
        X_dropped, y_dropped = self._remove_outliers(Outlier_indexes, X, y, verbose=self._verbose, visualize=True)

        if y_dropped is not None:
            return X_dropped, y_dropped
        else:
            return X_dropped
