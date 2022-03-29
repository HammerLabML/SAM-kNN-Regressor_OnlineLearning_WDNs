import numpy as np
import math
import sklearn.neighbors as sk
import time
from pykdtree.kdtree import KDTree
from skmultiflow.core import RegressorMixin


class SAMKNNRegressor(RegressorMixin):

    def __init__(self, n_neighbors=5, max_LTM_size=50, leaf_size=30, LTM_clean_strictness=0.5, multi_dim_y = True, error_type = 'RMSE'):
        self.n_neighbors = n_neighbors  # k
        self.max_LTM_size = max_LTM_size  # LTM size
        self.STMX, self.STMy, self.LTMX, self.LTMy = ([], [], [], [])
        self.STMerror, self.LTMerror, self.COMBerror, self.modelError, self.sampleCount = (0, 0, 0, 0, 0)
        self.leaf_size = leaf_size
        self.LTM_clean_strictness = LTM_clean_strictness
        self.multi_dim_y = multi_dim_y
        self.error_type = error_type
        self.adaptions = 0
        self.best_mem = -1

        # self.window = InstanceWindow(max_size=max_window_size, dtype=float)

    def partial_fit(self, X, y, sample_weight=None):
        """ Partially (incrementally) fit the model.
        Parameters
        ----------
        X : numpy.ndarray of shape (n_samples, n_features)
            The features to train the model.
        y: numpy.ndarray of shape (n_samples)
            An array-like with the target values of all samples in X.
        sample_weight: numpy.ndarray of shape (n_samples), optional (default=None)
            Samples weight. If not provided, uniform weights are assumed. Usage varies depending on the learning method.
        Returns
        -------
        self
        """

        r = X.shape[0]
        for i in range(r):
            self._partial_fit(X[i, :], y[i])

    def _partial_fit(self, x, y):
        self.STMX.append(x)
        self.STMy.append(y)
        self.sampleCount += 1

        # build up initial LTM
        if len(self.LTMX) < self.n_neighbors:
            self.LTMX.append(x)
            self.LTMy.append(y)

        if len(self.STMX) < self.n_neighbors:
            return

        self._evaluateMemories(x, y)
        self._adaptSTM()
        self._cleanLTM(x, y)
        self._enforceMaxLTMSize()

    def _enforceMaxLTMSize(self):

        while (len(self.LTMX) > self.max_LTM_size):
            ltm_tree = KDTree(np.array(self.LTMX))
            dist, ind = ltm_tree.query(np.array(self.LTMX), k=2)
            dist = dist[:, 1]
            ind = ind[:, 1]

            min_idx = np.argmin(dist)
            idx = ind[min_idx]

            self.LTMX = np.delete(self.LTMX, idx, axis=0).tolist()
            self.LTMy = np.delete(self.LTMy, idx, axis=0).tolist()


    def _cleanDiscarded(self, discarded_X, discarded_y):
        stm_tree = KDTree(np.array(self.STMX))
        STMy = np.array(self.STMy)

        clean_mask = np.zeros(discarded_X.shape[0], dtype=bool)

        for x, y in zip(self.STMX, self.STMy):
            # searching for points from the stm in the stm will also return that points, so we query one more to get k neighbours
            dist, ind = stm_tree.query(np.array([x]), k=self.n_neighbors + 1)
            dist = dist[0]
            ind = ind[0]

            """
            find weighted maximum difference and max distance among next n neighbours in STM
            """
            dist_max = np.amax(dist)

            if (self.multi_dim_y == False):
                w_diff = self._clean_metric(STMy[ind] - y, dist, dist_max)
                w_diff_max = np.amax(w_diff)
            else:
                w_diff = self._clean_metric(np.sum(STMy[ind] - y, axis=1) / np.array(y).shape[0], dist, dist_max)
                w_diff_max = np.amax(w_diff)

            """
            Query all points among the discarded that lie inside the maximum distance. Delete every point that has a greater weighted difference than the previously gathered maximum distance.
            """
            discarded_tree = KDTree(discarded_X)

            dist, ind = discarded_tree.query(
                np.array([x]),
                k=len(discarded_X),
                distance_upper_bound=dist_max)

            keep = ind < len(discarded_X)
            ind = ind[keep]
            dist = dist[keep]

            if (self.multi_dim_y == False):
                disc_w_diff = self._clean_metric(discarded_y[ind] - y, dist, dist_max)
            else:
                disc_w_diff = self._clean_metric(np.sum(discarded_y[ind] - y, axis=1) / np.array(y).shape[0], dist, dist_max)

            clean = ind[disc_w_diff < w_diff_max]

            """
            We create a mask which us used to drop all values in the discarded
            set whose weighted difference is to far from __all__ points.
            E.g. it does not appear in neighbourhood of any other point or
            is too different if it does.
            """

            clean_mask[clean] = True

        discarded_X = discarded_X[clean_mask]
        discarded_y = discarded_y[clean_mask]

        return discarded_X, discarded_y

    def _clean_metric(self, diffs, dists, norm=1.):
        # inverse distance weighting

        return np.abs(diffs) * 1 / np.exp(dists / (norm))


    def _cleanLTM(self, x, y):
        LTMX = np.array(self.LTMX)
        LTMy = np.array(self.LTMy)
        STMX = np.array(self.STMX)
        STMy = np.array(self.STMy)

        stmtree = KDTree(STMX)  # , self.leaf_size, metric='euclidean')

        dist, ind = stmtree.query(np.array([x]), k=self.n_neighbors)
        dist = dist[0]  # only queriing one point
        ind = ind[0]  # ^

        # print(dist)
        dist_max = np.amax(dist)

        if (self.multi_dim_y == False):
            qs = self._clean_metric(STMy[ind] - y, dist, dist_max)
            w_diff_max = np.amax(qs)
        else:
            qs = self._clean_metric(np.sum(STMy[ind] - y, axis=1) / y.shape[0], dist, dist_max)
            w_diff_max = np.amax(qs)

        ltmtree = KDTree(LTMX)
        dist, ind = ltmtree.query(
            np.array([x]),
            k=len(LTMX),
            distance_upper_bound=dist_max)

        keep = ind < len(LTMX)
        ind = ind[keep]
        dist = dist[keep]

        if (self.multi_dim_y == False):
            qstest = self.LTM_clean_strictness * self._clean_metric(LTMy[ind] - y, dist, dist_max)
            dirty = ind[qstest > w_diff_max]
        else:
            qstest = self.LTM_clean_strictness * self._clean_metric(np.sum(LTMy[ind] - y, axis=1) / y.shape[0], dist, dist_max)
            dirty = ind[qstest > w_diff_max]

        if (dirty.size):
            if (LTMX.shape[0] - len(dirty) < 5):
                return
            self.LTMX = np.delete(LTMX, dirty, axis=0).tolist()
            self.LTMy = np.delete(LTMy, dirty, axis=0).tolist()



    def _predict(self, X, y, x):
        X = np.array(X)
        y = np.array(y)

        tree = KDTree(X, X.shape[1])
        dist, ind = tree.query(np.array([x]), k=self.n_neighbors)

        dist = dist[0]
        ind = ind[0]

        clean = np.nonzero(dist)
        dist = dist[clean]
        ind = ind[clean]
        if len(dist) == 0:
            return 0

        if (self.multi_dim_y == False):
            pred = np.sum(y[ind] / dist)
            norm = np.sum(1 / dist)
        else:
            list = []
            for i in range(y[ind].shape[0]):
                list.append(y[ind][i] / dist[i])
            pred = np.sum(list, axis=0)
            norm = np.sum(1 / dist)

        if norm == 0:
            return 1

        return pred / norm

    def STMpredict(self, x):
        return self._predict(self.STMX, self.STMy, x)

    def LTMpredict(self, x):
        return self._predict(self.LTMX, self.LTMy, x)

    def COMBpredict(self, x):
        return self._predict(self.STMX + self.LTMX, self.STMy + self.LTMy, x)

    def _evaluateMemories(self, x, y):
        # absolute Mean Error Calculation
        if (self.error_type == 'MAE'):
            if(self.multi_dim_y == False):
                self.modelError = ((self.sampleCount - 1) * self.modelError + abs(self.predict([x]) - y)) / self.sampleCount
                STMsize = len(self.STMX)
                self.STMerror = ((STMsize - 1) * self.STMerror + abs(self.STMpredict(x) - y)) / STMsize
                LTMsize = len(self.LTMX)
                self.LTMerror = ((LTMsize - 1) * self.LTMerror + abs(self.LTMpredict(x) - y)) / LTMsize
                size = STMsize + LTMsize
                self.COMBerror = ((size - 1) * self.COMBerror + abs(self.COMBpredict(x) - y)) / size
            else:
                self.modelError = ((self.sampleCount - 1) * self.modelError + abs(np.sum(self.predict([x]) - y)) / y.shape[0]) / self.sampleCount
                STMsize = len(self.STMX)
                self.STMerror = ((STMsize - 1) * self.STMerror + abs(np.sum(self.STMpredict(x) - y)) / y.shape[0]) / STMsize
                LTMsize = len(self.LTMX)
                self.LTMerror = ((LTMsize - 1) * self.LTMerror + abs(np.sum(self.LTMpredict(x) - y)) / y.shape[0]) / LTMsize
                size = STMsize + LTMsize
                self.COMBerror = ((size - 1) * self.COMBerror + abs(np.sum(self.COMBpredict(x) - y)) / y.shape[0]) / size

        # Root Mean Squared Error Calculation
        elif (self.error_type == 'RMSE'):
            if (self.multi_dim_y == False):
                self.modelError = np.sqrt(((self.sampleCount - 1) * np.square(self.modelError) + np.square(self.predict([x]) - y)) / self.sampleCount)
                STMsize = len(self.STMX)
                self.STMerror = np.sqrt(((STMsize - 1) * np.square(self.STMerror) + np.square(self.STMpredict(x) - y)) / STMsize)
                LTMsize = len(self.LTMX)
                self.LTMerror = np.sqrt(((LTMsize - 1) * np.square(self.LTMerror) + np.square(self.LTMpredict(x) - y)) / LTMsize)
                size = STMsize + LTMsize
                self.COMBerror = np.sqrt(((size - 1) * np.square(self.COMBerror) + np.square(self.COMBpredict(x) - y)) / size)
            else:
                self.modelError = np.sqrt(((self.sampleCount - 1) * np.square(self.modelError) + np.square(np.sum(self.predict([x]) - y)) / y.shape[0]) / self.sampleCount)
                STMsize = len(self.STMX)
                self.STMerror = np.sqrt(((STMsize - 1) * np.square(self.STMerror) + np.square(np.sum(self.STMpredict(x) - y)) / y.shape[0]) / STMsize)
                LTMsize = len(self.LTMX)
                self.LTMerror = np.sqrt(((LTMsize - 1) * np.square(self.LTMerror) + np.square(np.sum(self.LTMpredict(x) - y)) / y.shape[0]) / LTMsize)
                size = STMsize + LTMsize
                self.COMBerror = np.sqrt(((size - 1) * np.square(self.COMBerror) + np.square(np.sum(self.COMBpredict(x) - y)) / y.shape[0]) / size)

    def _adaptSTM(self):
        STMX = np.array(self.STMX)
        STMy = np.array(self.STMy)

        best_MLE = self.STMerror
        best_size = STMX.shape[0]

        old_error = best_MLE
        old_size = best_size

        slice_size = int(STMX.shape[0] / 2)

        while (slice_size >= 50):
            MLE = 0

            for n in range(self.n_neighbors, slice_size):
                pred = self._predict(
                    STMX[-slice_size:-slice_size + n, :],
                    STMy[-slice_size:-slice_size + n],  # NOTE: multi dim y values possible?
                    STMX[-slice_size + n, :])

                if (self.error_type == 'MAE'):
                    if (self.multi_dim_y == False):
                        MLE += abs(pred - STMy[-slice_size + n])
                    else:
                        MLE += abs(np.sum(pred - STMy[-slice_size + n]) / pred.shape[0])
                elif (self.error_type == 'RMSE'):
                    if (self.multi_dim_y == False):
                        MLE += np.square(pred - STMy[-slice_size + n])
                    else:
                        MLE += np.square(np.sum(pred - STMy[-slice_size + n]) / pred.shape[0])

            if (self.error_type == 'MAE'):
                MLE = MLE / slice_size
            elif (self.error_type == 'RMSE'):
                MLE = np.sqrt(MLE / slice_size)

            if (MLE < best_MLE):
                best_MLE = MLE
                best_size = slice_size

            slice_size = int(slice_size / 2)

        if (old_size != best_size):
            self.adaptions += 1

            discarded_X = STMX[0:-best_size, :]
            discarded_y = STMy[0:-best_size]  # NOTE: multi dim y values possible?
            self.STMX = STMX[-best_size:, :].tolist()
            self.STMy = STMy[-best_size:].tolist()
            self.STMerror = best_MLE

            original_discard_size = len(discarded_X)

            # cleaning of the Discarded Set:
            discarded_X, discarded_y = self._cleanDiscarded(discarded_X, discarded_y)

            if (discarded_X.size):
                self.LTMX += discarded_X.tolist()
                self.LTMy += discarded_y.tolist()
            else:
                pass                

    def predict(self, X):
        """ predict

        Predicts the value of the X sample, by searching the KDTree for
        the n_neighbors-Nearest Neighbors.

        Parameters
        ----------
        X: Numpy.ndarray of shape (n_samples, n_features)
            All the samples we want to predict the label for.

        Returns
        -------
        list
            A list containing the predicted values for all instances in X.

        """

        mem_list = [self.STMerror, self.LTMerror, self.COMBerror]
        best_mem_ind = np.argmin(mem_list)
        if (best_mem_ind != self.best_mem):
            self.best_mem = best_mem_ind
        if (best_mem_ind == 0):
            return np.array([self.STMpredict(x) for x in X])
        elif (best_mem_ind == 1):
            return np.array([self.LTMpredict(x) for x in X])
        elif (best_mem_ind == 2):
            return np.array([self.COMBpredict(x) for x in X])

    def fit(self, X, y):
        if (len(self.STMX) < self.n_neighbors):
            self.LTMX = list(X[0:10, :])
            self.STMX = list(X[0:10, :])
            self.LTMy = list(y[0:10])
            self.STMy = list(y[0:10])
            self.sampleCount = 10
            self.partial_fit(X[10:, :], y[10:])
        else:
            self.partial_fit(X, y)

    def predict_proba(self):
        pass

    def print_model(self):
        print("Mean Absolute Errors:  Complete Model:", self.modelError, "STM: ", self.STMerror / len(self.STMX),
              "  LTM: ", self.LTMerror, "  COMB: ", self.COMBerror)