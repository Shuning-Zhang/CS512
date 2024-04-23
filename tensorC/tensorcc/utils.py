import sys
import os
import numpy as np
import logging as l
import datetime
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from algorithms.tensor_coclust_tau import CoClust


def execute_test(f, V, results, noise=0, method = 'ALT2',n_modes= 0):
    '''
    Execute CoClust algorithm and write an output file (already existing and open)

    Parameters:
    ----------

    f: output file (open). See CreateOutputFile for a description of the fields.      
    V: tensor
    results: classes on each mode (list of array, lists or tuples)
    noise: only for synthetic tensors. Amount of noise added to the perfect tensor
    method: variant of tauTCC
    n_modes: max number of modes of the tensors considered in the overall experiment

    '''
    
    
    model = CoClust(np.sum(V.shape) * 10, optimization_strategy = method)
    model.fit(V)
  
    l_nmi = []
    l_ari = []
    for i in range(len(results)):
        l_nmi.append(nmi(results[i], model._assignment[i], average_method='arithmetic'))
        l_ari.append(ari(results[i], model._assignment[i]))
    
    
    dim = ','.join(str(e) for e in V.shape)
    num_classes = ','.join(str(np.max(e) + 1) for e in results)
    tau = ','.join(str(e) for e in model.final_tau_)
    n = ','.join(str(e) for e in l_nmi)
    a = ','.join(str(e) for e in l_ari)
    num_clusters = ','.join(str(e) for e in model._n_clusters)
    if len(V.shape) < n_modes:
        c = ','*(n_modes-len(V.shape))
        dim += c
        num_classes += c
        tau += c
        n += c
        a += c
        num_clusters += c
            

    f.write(f"{dim},{num_classes},{noise},{tau},{n},{a},{num_clusters},{model.execution_time_},{method}\n")

    #f.write(f"{V.shape[0]},{V.shape[1]},{V.shape[2]},{np.max(x) + 1},{np.max(y) + 1},{np.max(z) + 1},{noise},{tau[0]},{tau[1]},{tau[2]},{nmi_x},{nmi_y},{nmi_z},{ari_x},{ari_y},{ari_z},{model._n_clusters[0]},{model._n_clusters[1]},{model._n_clusters[2]},{model.execution_time_},{sparsity}\n")


def CreateOutputFile(partial_name, own_directory = False, date = True, overwrite = False, modes = 3):
    '''
    Create and open a file containing the header described below.

    Parameters:
    ----------
    partial_name: partial name of the file and the directory that will contain the file.
    own_directory: boolean. Default: False.
        If true, a new directory './output/_{partial_name}/aaaa-mm-gg_hh.mm.ss' will be created.
        If flase, the path of the file will be './output/_{partial_name}'.
    date: boolean. Default: True.
        If true, the file name will include datetime.
        If false, it will not.
    overwrite: boolean. Default: False.
        If True, the pre-existing file with the same name is overwritten
    modes: integer. Default: 3.
        Max number of modes of the tensors considered in the overall experiment
                

    Output
    ------
    f: file (open). Each record contains the following fields, separated by commas (csv file):
        - dim_0,..., dim_n: dimension of the tensor on mode i
        - num_classes_0,...,num_classes_n: correct number of clusters on mode i
        - noise: only for synthetic tensors. Amount of noise added to the perfect tensor
        - tau_0,...,tau_n: final tau_i
        - nmi_0,...,nmi_n: normalized mutual information score on mode i
        - ari_0,...,ari_n: adjusted rand index on mode i
        - num_clusters_0, ..., num_clusters_n: number of clusters on mode i detected by tauTCC
        - execution time
        - method: optimization strategy of tauTCC adopted (one of ['ALT','ALT2','AGG2','AGG','AVG']

        File name:{partial_name}_aaaa-mm-gg_hh.mm.ss.csv or {partial_name}_results.csv
    dt: datetime (as in the directory/ file name)

    
    '''

    
    dt = f"{datetime.datetime.now()}"
    if own_directory:
        data_path = f"./output/_{partial_name}/" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + "/"
    else:
        data_path = f"./output/_{partial_name}/"
    directory = os.path.dirname(data_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    new = True
    if date:
        file_name = partial_name + "_" + dt[:10] + "_" + dt[11:13] + "." + dt[14:16] + "." + dt[17:19] + ".csv"
    else:
        file_name = partial_name + '_results.csv'
        if os.path.isfile(data_path + file_name):
            if overwrite:
                os.remove(data_path + file_name)
            else:
                new = False
            
            
    f = open(data_path + file_name, "a",1)
    if new:
        if modes == 3:
            f.write("dim_x,dim_y,dim_z,x_num_classes,y_num_classes,z_num_classes,noise,tau_x,tau_y,tau_z,nmi_x,nmi_y,nmi_z,ari_x,ari_y,ari_z,x_num_clusters,y_num_clusters,z_num_clusters,execution_time,method\n")
        else:
            l_dim = ["dim_" + str(i) for i in range(modes)]
            l_num_classes = ["num_classes_" + str(i) for i in range(modes)]
            l_tau = ["tau_" + str(i) for i in range(modes)]
            l_nmi = ["nmi_" + str(i) for i in range(modes)]
            l_ari = ["ari_" + str(i) for i in range(modes)]
            l_num_clusters = ["num_clusters_" + str(i) for i in range(modes)]

            dim = ','.join(e for e in l_dim)
            num_classes = ','.join(e for e in l_num_classes)
            tau = ','.join(e for e in l_tau)
            nmi = ','.join(e for e in l_nmi)
            ari = ','.join(e for e in l_ari)
            num_clusters = ','.join(e for e in l_num_clusters)

            
            
            f.write(f"{dim},{num_classes},noise,{tau},{nmi},{ari},{num_clusters},execution_time,method\n")

    return f, dt


def CreateLogger(input_level = 'INFO'):
    level = {'DEBUG':l.DEBUG, 'INFO':l.INFO, 'WARNING':l.WARNING, 'ERROR':l.ERROR, 'CRITICAL':l.CRITICAL}
    logger = l.getLogger()
    logger.setLevel(level[input_level])

    return logger
