import numpy as np
import river

from sklearn.linear_model import SGDRegressor


class KNN:

    def __init__(self, n_dim, k = 5, window_size = 500):

        x_0 = river.utils.numpy2dict(np.zeros(n_dim))

        self.model = river.neighbors.KNNRegressor(n_neighbors = k, window_size = window_size, aggregation_method='weighted_mean')
        self.model.learn_one(x_0, 0)

    def partial_predict(self, u_t):

        u_t = river.utils.numpy2dict(u_t)

        y_pred = self.model.predict_one(u_t)

        return y_pred

    def fit_sample(self, u_t, y_t):

        u_t = river.utils.numpy2dict(u_t)

        self.model.learn_one(u_t, y_t)

    def predict(self, U):

        y_pred = []
        for i in range(U.shape[0]):

            y_pred.append(self.partial_predict(U[i,:]))

        return np.asarray(y_pred)

    def partial_fit(self, U, Y):

        for i in range(U.shape[0]):

            self.fit_sample(U[i,:], Y[i])


class LinearModel:

    def __init__(self, n_dim):

        x_0 = river.utils.numpy2dict(np.zeros(n_dim))

        self.model = river.linear_model.LinearRegression()
        self.model.learn_one(x_0, 0)

    def partial_predict(self, u_t):

        u_t = river.utils.numpy2dict(u_t)

        y_pred = self.model.predict_one(u_t)

        return y_pred

    def fit_sample(self, u_t, y_t):

        u_t = river.utils.numpy2dict(u_t)

        self.model.learn_one(u_t, y_t)

    def predict(self, U):

        y_pred = []
        for i in range(U.shape[0]):
            y_pred.append(self.partial_predict(U[i, :]))

        return np.asarray(y_pred)

    def partial_fit(self, U, Y):

        for i in range(U.shape[0]):
            self.fit_sample(U[i, :], Y[i])
