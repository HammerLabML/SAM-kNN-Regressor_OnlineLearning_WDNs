mkdir exexperiments_results

python experiments.py 1 knn False > experiments_results/scenario-1_knn_leakage_onlinelearning.txt
python experiments.py 2 knn False > experiments_results/scenario-2_knn_leakage_onlinelearning.txt
python experiments.py 3 knn False > experiments_results/scenario-3_knn_leakage_onlinelearning.txt
python experiments.py 4 knn False > experiments_results/scenario-4_knn_leakage_onlinelearning.txt
python experiments.py 5 knn False > experiments_results/scenario-5_knn_leakage_onlinelearning.txt

python experiments.py 1 linear False > experiments_results/scenario-1_linear_leakage_onlinelearning.txt
python experiments.py 2 linear False > experiments_results/scenario-2_linear_leakage_onlinelearning.txt
python experiments.py 3 linear False > experiments_results/scenario-3_linear_leakage_onlinelearning.txt
python experiments.py 4 linear False > experiments_results/scenario-4_linear_leakage_onlinelearning.txt
python experiments.py 5 linear False > experiments_results/scenario-5_linear_leakage_onlinelearning.txt

python experiments.py 1 samknn False > experiments_results/scenario-1_samknn_leakage_onlinelearning.txt
python experiments.py 2 samknn False > experiments_results/scenario-2_samknn_leakage_onlinelearning.txt
python experiments.py 3 samknn False > experiments_results/scenario-3_samknn_leakage_onlinelearning.txt
python experiments.py 4 samknn False > experiments_results/scenario-4_samknn_leakage_onlinelearning.txt
python experiments.py 5 samknn False > experiments_results/scenario-5_samknn_leakage_onlinelearning.txt

python experiments.py 1 knn True > experiments_results/scenario-1_knn_sensorfault_onlinelearning.txt
python experiments.py 2 knn True > experiments_results/scenario-2_knn_sensorfault_onlinelearning.txt
python experiments.py 3 knn True > experiments_results/scenario-3_knn_sensorfault_onlinelearning.txt
python experiments.py 4 knn True > experiments_results/scenario-4_knn_sensorfault_onlinelearning.txt
python experiments.py 5 knn True > experiments_results/scenario-5_knn_sensorfault_onlinelearning.txt

python experiments.py 1 linear True > experiments_results/scenario-1_linear_sensorfault_onlinelearning.txt
python experiments.py 2 linear True > experiments_results/scenario-2_linear_sensorfault_onlinelearning.txt
python experiments.py 3 linear True > experiments_results/scenario-3_linear_sensorfault_onlinelearning.txt
python experiments.py 4 linear True > experiments_results/scenario-4_linear_sensorfault_onlinelearning.txt
python experiments.py 5 linear True > experiments_results/scenario-5_linear_sensorfault_onlinelearning.txt

python experiments.py 1 samknn True > experiments_results/scenario-1_samknn_sensorfault_onlinelearning.txt
python experiments.py 2 samknn True > experiments_results/scenario-2_samknn_sensorfault_onlinelearning.txt
python experiments.py 3 samknn True > experiments_results/scenario-3_samknn_sensorfault_onlinelearning.txt
python experiments.py 4 samknn True > experiments_results/scenario-4_samknn_sensorfault_onlinelearning.txt
python experiments.py 5 samknn True > experiments_results/scenario-5_samknn_sensorfault_onlinelearning.txt
