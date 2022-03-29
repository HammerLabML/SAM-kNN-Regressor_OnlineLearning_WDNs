import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from SAM_KNN_Regression import SAMKNNRegressor


class MyModel():
    def __init__(self, model=SAMKNNRegressor(multi_dim_y=False, max_LTM_size=100, n_neighbors=5, error_type="MAE")):
        self.model = model
    
    def fit(self, X, y):
        self.model.partial_fit(X, y)
    
    def predict(self, X):
        return self.model.predict(X)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return r2_score(y, y_pred)
        #return mean_squared_error(y, y_pred)
        #return mean_absolute_error(y, y_pred)
    
    def get_stat_abs_deviation(self, X, y):
        y_pred = self.predict(X)
        difference_abs = np.abs(y_pred - y)
        
        return np.max(difference_abs), np.min(difference_abs), np.mean(difference_abs)


def create_model(model):
    return MyModel(model)
