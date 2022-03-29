import sys
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from LeakDbScenarioPreprocessing import global_preprocessing
from MyModel import MyModel, create_model
from FaultDetection import EnsembleSystem
from utils import evaluate_fault_detection
from models import KNN, LinearModel
from SAM_KNN_Regression import SAMKNNRegressor


path_to_data = "data/"

time_win_len = 4  # Always consider the last four time steps
time_start = 100  # Ignore the first 100 samples

t_train_split = 3000    # Use the first 3000 sampels for training -- there are no anomalies present early until end of January!  
batch_size = 200  # Number of samples used for adapting the model to the fault


def load_data(scenario_id, use_faulty_sensor_scenario=False):
    # Open and read file
    df_pressures = pd.read_csv(os.path.join(path_to_data, f"Scenario-{scenario_id}/Results/Measurements_Pressures.csv"))
    df_flows = pd.read_csv(os.path.join(path_to_data, f"Scenario-{scenario_id}/Results/Measurements_Flows.csv"))
    
    # Parse labels
    df_labels = df_pressures[["Timestamp"]].copy()
    df_labels["label"] = 0

    def parse_leak_data(): # TODO: Assuming there is only one leaky node!
        leaky_pipe, start, end = None, None, None
        
        # Locate and load file with information on the leak
        leak_info_file = list(filter(lambda z: z.endswith(".xlsx"), os.listdir(os.path.join(path_to_data, f"Scenario-{scenario_id}/ResultsLeakages/"))))[0]
        df_leak_info = pd.read_excel(os.path.join(path_to_data, f"Scenario-{scenario_id}/ResultsLeakages/{leak_info_file}"), sheet_name="Info", engine='openpyxl')
        
        # Parse for start and end time of leak
        for _, row in df_leak_info.iterrows():
            if row["Description"] == "Leak Start":
                start = row["Value"]
            elif row["Description"] == "Leak End":
                end = row["Value"]
            elif row["Description"] == "Leak Pipe":
                leaky_pipe = row["Value"]
        
        return leaky_pipe, start, end

    leaky_pipe, start, end = parse_leak_data()

    # Load faulty sensor information
    def load_faulty_sensor_info():
        faulty_sensor_id, start, end = None, None, None

        faulty_sensor_info_file = list(filter(lambda z: z.endswith(".csv"), os.listdir(os.path.join(path_to_data, f"Scenario-{scenario_id}/ResultsSensorFaults/WithoutSensorFaults/"))))[0]
        df_faulty_sensor_info = pd.read_csv(os.path.join(path_to_data, f"Scenario-{scenario_id}/ResultsSensorFaults/WithoutSensorFaults/{faulty_sensor_info_file}"))
        for _, row in df_faulty_sensor_info.iterrows():
            if row["Description"] == "Sensor":
                faulty_sensor_id = row["Value"].replace("node", "") # Remove naming artificat from the generator
            elif row["Description"] == "Fault Start":
                start = row["Value"]
            elif row["Description"] == "Fault End":
                end = row["Value"]

        return faulty_sensor_id, start, end
    
    faulty_sensor_id = None
    if use_faulty_sensor_scenario is True:
        faulty_sensor_id, start, end = load_faulty_sensor_info()    # Note: We override start and end time of the leakage

    if use_faulty_sensor_scenario is True:  # Data from which simulation should be used?
        df_pressures = pd.read_csv(os.path.join(path_to_data, f"Scenario-{scenario_id}/ResultsSensorFaults/Measurements_Pressures.csv"))
        df_flows = pd.read_csv(os.path.join(path_to_data, f"Scenario-{scenario_id}/ResultsSensorFaults/Measurements_Flows.csv"))

    # Parse labels
    indices = df_labels[(df_labels["Timestamp"] >= start) & (df_labels["Timestamp"] <= end)].index
    for idx in indices:
        df_labels["label"].loc[idx] = 1

    labels = df_labels["label"].to_numpy().flatten()

    # Parse pressure and flow measurements
    df_pressures.drop(columns=["Timestamp"], inplace=True)
    df_flows.drop(columns=["Timestamp"], inplace=True)

    pressure_nodes = list(df_pressures.columns)
    flow_nodes = list(df_flows.columns)
    nodes = pressure_nodes + flow_nodes

    pressures_per_node = {}
    for node_id in pressure_nodes:
        pressures_per_node[node_id] = df_pressures[[node_id]].to_numpy().flatten()

    flows_per_node = {}
    for node_id in flow_nodes:
        flows_per_node[node_id] = df_flows[[node_id]].to_numpy().flatten()

    # Build numpy arrays
    y = labels
    X = np.vstack([pressures_per_node[n] for n in pressure_nodes] + [flows_per_node[n] for n in flow_nodes]).T

    return X, y, nodes, pressure_nodes, flow_nodes, leaky_pipe, faulty_sensor_id


def score(Y_True, Y_pred):
    Y_pred = Y_pred.reshape(Y_True.shape)
    return [mean_squared_error(Y_True[:, i], Y_pred[:, i]) for i in range(Y_True.shape[1])]


if __name__ == "__main__":
    # Parse arguments
    if len(sys.argv) != 4:
        print("Usage: <scenario-id> <model-desc> <use_faulty_sensor_scenario>")
        os._exit(1)
    
    scenario_id = sys.argv[1]
    model_desc = sys.argv[2]
    use_faulty_sensor_scenario = True if sys.argv[3] == "True" else False

    # Load data
    X, y, nodes, pressure_nodes, flow_nodes, leaky_pipe, faulty_sensor = load_data(scenario_id, use_faulty_sensor_scenario)

    # Preprocessing
    X_all, Y_all, leak_labels_all = global_preprocessing(X, y, time_start, time_win_len, list(range(len(pressure_nodes))))  
    n_dim = X_all.shape[1]

    X_all_train, X_all_test = X_all[:t_train_split,:], X_all[t_train_split:,:]
    Y_all_train, Y_all_test = Y_all[:t_train_split,:], Y_all[t_train_split:,:]
    fault_labels_all_train, fault_labels_all_test = leak_labels_all[:t_train_split], leak_labels_all[t_train_split:]

    faulty_times = np.where(fault_labels_all_test == 1)[0]
    test_t0, test_t1= faulty_times[0], faulty_times[-1] + 50  # Time period in the test set where a fault is present + some a time buffer for the original hydraulics to recover from the fault

    # Fit model
    def get_model():
        if model_desc == "knn":
            return KNN(n_dim-1, k=5, window_size=500)
        elif model_desc == "linear":
            return LinearModel(n_dim-1)
        elif model_desc == "samknn":
            return SAMKNNRegressor(multi_dim_y=False, max_LTM_size=100, n_neighbors=5, error_type="MAE")

    my_model_gen = lambda: create_model(get_model())
    model = EnsembleSystem(my_model_gen, flow_nodes, pressure_nodes)
    #model = EnsembleSystem(MyModel, flow_nodes, pressure_nodes)
    model.fit(X_all_train, Y_all_train)

    # Online learning -- test if anomalies are found
    suspicous_time_points = []
    t = 0
    while t + batch_size < X_all_test.shape[0]:
        X_batch, Y_batch = X_all_test[t:t+batch_size,:], Y_all_test[t:t+batch_size,:]

        # Apply anomaly detection to current batch
        time_points, _ = model.apply_detector(X_batch, Y_batch)
        for tp in time_points:
            suspicous_time_points.append(t + tp)

        # Adapt to current batch
        model.fit(X_batch, Y_batch)

        # Next batch
        t += batch_size

    # Evaluation
    score_fault_detection = evaluate_fault_detection(suspicous_time_points, faulty_times)
    print(score_fault_detection)

    # Store scores
    scenario_config = "faultysensor" if use_faulty_sensor_scenario is True else "leakage"
    np.savez(f"experiments_results/scenario-{scenario_id}_{scenario_config}_{model_desc}_onlinelearning.npz", feature_desc=pressure_nodes, score_fault_detection=score_fault_detection, suspicous_time_points=suspicous_time_points, faulty_times=faulty_times)
