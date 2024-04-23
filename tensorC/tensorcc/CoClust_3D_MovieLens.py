import pandas as pd
import numpy as np
import os
import sys
import json
from algorithms.tensor_coclust_tau import CoClust
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
from utils import CreateOutputFile, CreateLogger

if len(sys.argv)> 1 and sys.argv[1] == '-h':
    print('''
    Co-Clustering of tensors extracted from MovieLens database. Parameters:
        tensor: integer.
            Accepted values: 1 or 2.
        algorithm: string. Optional.
            Accepted values: ALT, AVG, AGG, ALT2, AGG2. Default: ALT2
        level: string. Optional.
            Accepted values: DEBUG, INFO, WARNING, ERROR, CRITICAL
            Default value: WARNING
    ''')
    quit()

# k = sys.argv[1]
alg = sys.argv[2] if (len(sys.argv)> 2) and (sys.argv[2] in ['ALT','AVG','AGG','ALT2','AGG2']) else 'ALT2'
level = sys.argv[-1] if (len(sys.argv)> 2) and (sys.argv[-1] in ['DEBUG','INFO','WARNING','ERROR','CRITICAL']) else 'WARNING'
logger = CreateLogger(level)

f, dt = CreateOutputFile("MovieLens", date = True)

output_path = f"./output/_MovieLens/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
directory = os.path.dirname(output_path)
if not os.path.exists(directory):
    os.makedirs(directory)

# final = pd.read_pickle('./resources/movielens/movielens_final_'+ k + '.pkl')
# n = np.shape(final.groupby('userID').count())[0]
# m = np.shape(final.groupby('movieID').count())[0]
# l = np.shape(final.groupby('tagID').count())[0]
# T = np.zeros((n,m,l))
# y = np.zeros(m)
# for index, row in final.iterrows():
#     T[row['user_le'], row['movie_le'], row['tag_le']] = 1
#     y[row['movie_le']] = row['genre_le']
test_idx = pd.read_csv('../../dataset/cc dataset/test_idx.txt', delimiter='\t',header=None)
output_data = []
file_name = '../../dataset/cc dataset/costco_result-2.csv'
with open(file_name, 'rb') as f:
        output_data = json.load(f)

n = np.shape(test_idx.groupby(0).count())[0]
m = np.shape(test_idx.groupby(1).count())[0]
T = np.zeros((610,m))
y = np.zeros(m)
for index, row in test_idx.iterrows():
    T[row[0], row[1]% m] = 1
    y[row[1]%m] = output_data[index]

#sparsity = 1 - (np.sum(T>0) / np.product(T.shape))

model = CoClust(n_iterations = np.sum(T.shape) * 100, optimization_strategy = alg, path = output_path)
model.fit(T)

tau = model.final_tau_
n = nmi(model._assignment[1], y, average_method='arithmetic')
a = ari(model._assignment[1], y)
acc = accuracy_score(model._assignment[1], y)
print('n:', n, 'a:', a,'accuracty:', acc)

f.write(f"{T.shape[0]},{T.shape[1]},{T.shape[2]},,{len(set(y))},,,{tau[0]},{tau[1]},{tau[2]},,{n},,,{a},,{model._n_clusters[0]},{model._n_clusters[1]},{model._n_clusters[2]},{model.execution_time_},{alg}\n")
f.close()

gy = open(output_path + alg + "_assignments_ML_y.txt", 'w')
for i in range(T.shape[1]):
    gy.write(f"{i}\t{model._assignment[1][i]}\n")

gy.close()

gz = open(output_path + alg + "_assignments_ML_z.txt", 'w')
for i in range(T.shape[2]):
    gz.write(f"{i}\t{model._assignment[2][i]}\n")

gz.close()
