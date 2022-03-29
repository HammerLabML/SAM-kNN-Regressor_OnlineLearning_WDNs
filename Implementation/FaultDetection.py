import numpy as np



class FaultDetection():
    def __init__(self, model, threshold):
        self.model = model
        self.threshold = threshold
    
    def fit(self, X_train, Y_train):
        self.model.fit(X_train, Y_train)

    def score(self, X, Y):
        return self.model.score(X, Y)

    def apply_detector(self, X, Y):
        return self.apply_to_labeled_stream(X, Y), None

    def apply_to_labeled_stream(self, X, y):
        y_pred = self.model.predict(X)
        
        return list(np.where(np.abs(y_pred - y) > self.threshold)[0])


def compute_indicator_function(time_points, y_leaks):
    return np.array([1. if t in time_points else 0. for t in range(len(y_leaks))])


class EnsembleSystem():
    def __init__(self, model_class, flow_nodes, pressure_nodes, fault_detecotor_class=FaultDetection):
        self.model_class = model_class
        self.fault_detecotor_class = fault_detecotor_class
        #self.flow_nodes = flow_nodes
        self.pressure_nodes = pressure_nodes    # Only pressure nodes are used!
        self.models = []
    
    def fit(self, X_train, Y_train):    # This is basically a partial_fit
        if len(self.models) == 0:
            self.models = [{"model": self.model_class()} for _ in range(len(self.pressure_nodes))]
        
        for i in range(len(self.pressure_nodes)):
            # Select inputs and output
            inputs_idx = list(range(X_train.shape[1]));inputs_idx.remove(i)
            x_train, y_train = X_train[:,inputs_idx], Y_train[:,i]
        
            # Fit regression
            model = self.models[i]["model"]
            model.fit(x_train, y_train)
            
            # Build fault detector
            max_abs_error = 1.5 * model.get_stat_abs_deviation(x_train, y_train)[0]   # TODO: Magic number
            fault_detector = self.fault_detecotor_class(model, max_abs_error)
            
            # Store model
            self.models[i] = {"model": model, "fault_detector": fault_detector, "input_idx": inputs_idx, "target_idx": i}

    def predict(self, X):
        Y = []
        
        for m in self.models:
            Y.append(m["model"].predict(X[:,m["input_idx"]]))
            
        Y = np.array(Y).T
        return Y
    
    def score(self, X, Y):
        scores = []
        
        for m in self.models:
            scores.append(m["model"].score(X[:,m["input_idx"]], Y[:,m["target_idx"]]))
        
        return scores
    
    def apply_detector(self, X, Y):
        suspicious_time_points = []
        sensor_forecasting_errors = []
        
        for m in self.models:
            x_in = X[:,m["input_idx"]]
            y_truth = Y[:,m["target_idx"]]
            
            sensor_forecasting_errors.append(np.square(m["model"].predict(x_in) - y_truth))# Squared error
            suspicious_time_points += m["fault_detector"].apply_to_labeled_stream(x_in, y_truth)
        
        sensor_forecasting_errors = np.vstack(sensor_forecasting_errors).T
        suspicious_time_points = list(set(suspicious_time_points));suspicious_time_points.sort()   # TODO: Majority voting, weighting, ...?
        return suspicious_time_points, sensor_forecasting_errors
