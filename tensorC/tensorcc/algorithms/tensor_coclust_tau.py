import logging
from random import choice
from random import randint
from time import time
from typing import List

import numpy as np
import scipy
from itertools import product
from scipy.sparse import issparse
from sklearn.base import BaseEstimator
from sklearn.base import ClusterMixin
from sklearn.base import TransformerMixin
from sklearn.utils import check_array


class CoClust(BaseEstimator, ClusterMixin, TransformerMixin):
    """
    CoClust is a co-clustering algorithm created to deal with 3D-tensor data.
    It finds automatically the best number of row / column / slice clusters.

    Parameters
    ------------

    n_iterations : int, optional, default: 300
        The number of iterations to be performed.

    optimization strategy: {'ALT', 'AVG', 'AGG', 'ALT2', 'AGG2'}, optional, default: 'ALT2'
        The version of the algorithm. The possible values are:
            

    min_n_x_clusters : int, optional, default: 2,
        The number of clusters for the first mode, should be at least 2

    min_n_y_clusters : int, optional, default: 2,
        The number of clusters for the second mode, should be at least 2

    min_n_z_clusters : int, optional, default: 2,
        The number of clusters for the third mode, should be at least 2

    n_threshold : int, optional, default: 0,
        The number of iterations without moves after which the object selection strategy changes.
        If 0, the number is set equal to the dimension of the highest-dimensional mode

    path : string, optional, default: None,
        If a path is specified, the algorithm prints in the specified path a file every 100 iterations containing
        the partial results.

    compute_tau_list : boolean, optional, default : False,
        If True, the value of taus after each iteration is saved in a parameter "tau_vector_".
    

    Attributes
    -----------

    x_ : array, length dimension on the first mode
        Results of the clustering on x mode. `x_[i]` is `c` if
        object `i` is assigned to cluster `c`. Available only after calling ``fit``.

    y_ : array, length dimension on the second mode
        Results of the clustering on y mode. `y_[i]` is `c` if
        object `i` is assigned to cluster `c`. Available only after calling ``fit``.

    z_ : array, length dimension on the third mode
        Results of the clustering on z mode. `z_[i]` is `c` if
        object `i` is assigned to cluster `c`. Available only after calling ``fit``.

    execution_time_ : float
        The execution time.

    initial_tau_ : tuple, length 3
        Initial values of tau_x, tau_y, tau_z, when the co-clustering is
        the discrete one. Available only after calling ``fit``.

    final_tau_ : tuple, length 3
        Values of tau_x, tau_y, tau_z corresponding to the final
        co-clustering solution. Available only after calling ``fit``.

    tau_vector_ : list, length actual number of iterations
        values of (tau_x, tau_y, tau_Z) after each iteration. Available only if compute_tau_list == True.



    References
    ----------

    * Battaglia, Pensa: A parameter-less algorithm for tensor co-clustering.

    """

    def __init__(self, n_iterations=500, optimization_strategy = 'ALT2', min_n_clusters=2, n_threshold = 0, path = None, compute_tau_list = False):
        """
        Create the model object and initialize the required parameters.

        :type n_iterations: int
        :param n_iterations: the max number of iterations to perform
        :type optimization_strategy: string, one of {'ALT', 'AVG', 'AGG', 'ALT2', 'AGG2'}
        :param optimization_strategy: version of the algorithm
        :type min_n_x_clusters, min_y_clusters, min_z_clusters: int
        :param min_n_x_clusters, min_n_y_clusters, min_n_z_clusters: the minimum number of clusters (per mode)
        :type n_threshold: int
        :param n_threshold: max number of iterations without any move before changing object selection strategy
        :type path: string
        :param path: if not None, path where to save intermediate results
        :type compute_tau_list: boolean
        :param compute_tau_list: if True, save the value of tau function after each iteration
        """

        self.n_iterations = n_iterations
        self.min_n_clusters = min_n_clusters
        self.n_threshold = n_threshold
        self.path = path
        self.compute_tau_list = compute_tau_list
        self.optimization_strategy = optimization_strategy

        # these fields will be available after calling fit
        #self.x_ = None
        #self.y_ = None
        #self.z_ = None
        self.n_clusters_ = None

        self.initial_tau_ = None
        self.final_tau_ = None

        np.seterr(all='ignore')

    def _init_all(self, V):
        """
        Initialize all variables needed by the model.

        :param V: the dataset
        :return:
        """
        # verify that all matrices are correctly represented
        # check_array is a sklearn utility method
        self._dataset = None

        self._dataset = check_array(V, accept_sparse='csr', ensure_2d = False, allow_nd = True, dtype=[np.int32, np.int8, np.float64, np.float32])
        logging.debug("[INFO] dataset done")

        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            
        # final number of iterations
        self._last_iteration = 0
        
        # dimensions in each mode
        self._n = self._dataset.shape
        self._n_modes = len(self._dataset.shape)

        # set the default value of the threshold to max(dimension)
        if self.n_threshold == 0:
            self.n_threshold = np.max(self._n)

        # the number of clusters on each mode
        self._n_clusters = np.zeros(self._n_modes, dtype = 'int16')

        # for each mode, an array containing the number of cluster to which each element has been assigned
        self._assignment = [np.zeros(self._n[i], 'int16')for i in range(self._n_modes)]

        # T is the contingency tensor
        self._T = None

        # computation time
        self.execution_time_ = 0

	#(tau_x, tau_y, tau_z)
        #self.initial_tau_ = (0,0,0)
        #self.final_tau_ = (0,0,0)

        # number of performed moves on each mode
        self._performed_moves = np.zeros(self._n_modes, dtype = int)

        # number of iterations with no moves
        self._iterations_without_moves = 0
        
        # initialize all 
        self._discrete_initialization()
        logging.debug("[INFO] initialization done")

        self._update_intermediate_values_after_move(self._n_modes -1)
        logging.debug("[INFO] intermediate values done")

        


        ############

        #self._tau = np.zeros(self._n_modes)        
        self.tau_vector_ = list()

        
        #logging.debug("[INFO] X's initialization: {0}".format(list(self._assignment[0])))
        #logging.debug("[INFO] Y's initialization: {0}".format(list(self._assignment[1])))
        #logging.debug("[INFO] Z's initialization: {0}".format(list(self._assignment[2])))
        logging.debug("[INFO] initialization: {0}".format(list(self._assignment)))



    def fit(self, V, y=None):
        """
        Fit CoClust to the provided data.

        Parameters
        -----------

        V : np.array with three modes

        y : unused parameter

        Returns
        --------

        self

        """

        # Initialization phase
        self._init_all(V)

        start_time = time()

        # Execution phase
        actual_n_iterations = 0

        #self.initial_tau_ = self._compute_taus()    

        while actual_n_iterations != self.n_iterations:
            logging.debug("[INFO] iteration {0}".format(actual_n_iterations))
            
            # each iterations performs a move on rows and n_view moves on columns

            iter_start_time = time()
            logging.debug("[INFO] ##############################################\n" +
                         "\t\t### Iteration {0}".format(actual_n_iterations))
            #logging.debug("[INFO] X assignment: {0}".format(self._assignment[0]))
            #logging.debug("[INFO] Y assignment: {0}".format(self._assignment[1]))
            #logging.debug("[INFO] Z assignment: {0}".format(self._assignment[2]))


            # perform a move in each dimension
            for i in range(self._n_modes):
                logging.debug("[INFO] Dimension {0}".format(i))
                
                if self._iterations_without_moves == self._n_modes * self.n_threshold:
                    self._actual_item = [randint(1, self._n[j] * 100) % self._n[j] for j in range(self._n_modes)]
                    self._special_case = 0
                elif self._iterations_without_moves > self._n_modes * self.n_threshold:
                    self._special_case += 1
                else:
                    self._special_case = -1
                    
                if int(self._special_case / self._n_modes) < self._n[i]:
                    self._perform_move(i)
                else:
                    self._update_intermediate_values_after_move(i)
                
            if self.compute_tau_list:
                tau = self._compute_taus()
                self.tau_vector_.append(tau)
                
            iter_end_time = time()

            
            if actual_n_iterations % 1 == 0:
                logging.info("Iteration #{0}".format(actual_n_iterations))
                logging.info("[INFO] # clusters: {0}".format(self._n_clusters))
                tau = self._compute_taus()
                logging.info("[INFO] Taus: {0}".format(tau))
                logging.info("[INFO] Iteration time: {0}".format(iter_end_time - iter_start_time))
            

            if self.path != None:
                if actual_n_iterations % 1000 == 0:
                    f = open(self.path + 'assignments_' + str(actual_n_iterations) + '.txt', 'w+')
                    
                    for j in range(self._n_modes):
                        f.write(f'##### mode {j} #####\n')
                        for i in range(self._n[j]):
                            f.write(f"{i}\t{self._assignment[j][i]}\n")
                    f.close()
                
            actual_n_iterations += 1
            if int(self._special_case / self._n_modes) == np.max(self._n):
                logging.info("[INFO] All moves have been tried. Taus can't be improved by moving any element.")
                self._last_iteration = actual_n_iterations
                actual_n_iterations = self.n_iterations

        end_time = time()

        execution_time = end_time - start_time
        self.final_tau_ = self._compute_taus()
        
        logging.info('#####################################################')
        logging.info("[INFO] Execution time: {0}".format(execution_time))

        # clone cluster assignments and transform in lists
        #self.x_ = np.copy(self._assignment[0]).tolist()
        #self.y_ = np.copy(self._assignment[1]).tolist()
        #self.z_ = np.copy(self._assignment[2]).tolist()
        self._n_clusters_ = np.copy(self._assignment)
        self.execution_time_ = execution_time

        logging.info("[INFO] Number of clusters found: {0}".format(self._n_clusters))
        for j in range(self._n_modes):
            logging.info("[INFO] Number of clusters found on mode {1}: {0}".format(self._n_clusters[j],j))
            logging.info("[INFO] Number of moves performed on mode {1}: {0} / {2}".format(self._performed_moves[j],j, self.n_iterations))

        return self

    def _discrete_initialization(self):
        """
        The initialization method assign each element on each mode to a new cluster.

        :param:

        :return:
        """

        # simply assign each element to a cluster in each mode
        for i in range(self._n_modes):
            self._n_clusters[i] = self._n[i]
            self._assignment[i] = np.arange(self._n[i], dtype='int16')
            

        #self._T = self._init_contingency_matrix()
        self._T = np.zeros(np.shape(self._dataset), dtype = 'int64') + self._dataset
                
        # init the T-derived fields for further computations
        self._init_T_derived_fields()

    """           
    def _init_contingency_matrix(self):
        
        #Initialize the T contingency tensor
        #of shape _n_clusters
        #For the moment the only initialization method available is the discrete one,
        #so the initial contingency tensor coincides to the input tensor V

        #:return: array with three modes
        
        logging.debug("[INFO] Compute the contingency matrix...")

        new_t = np.zeros(self._n_clusters, dtype=float)

        for di in range(self._n[0]):
            x_cluster_of_c = self._assignment[0][di]

            for ci in range(self._n[1]):
                y_cluster_of_c = self._assignment[1][ci]

                for ei in range(self._n[2]):
                    z_cluster_of_c = self._assignment[2][ei]
                    new_t[x_cluster_of_c][y_cluster_of_c][z_cluster_of_c] += self._dataset[di][ci][ei]


        logging.debug("[INFO] End of contingency matrix computation...")
        
        
        return new_t
    """
    
    def _init_T_derived_fields(self):
        """
        Initialize the vlaues that will be used in the computation of Delta tau

        :return:
        """

        logging.debug("[INFO] Init fields derived by the contingency matrix...")


        # tot_t_per_mode[j]: Vector (t1.., ..., tm..)
        # tot_t_other[j]: tensor of n-1 modes
        self._tot_t_per_mode = []
        self._tot_t_other = []
        for j in range(self._n_modes):
            axis = tuple([i for i in range(self._n_modes) if i != j])
            self._tot_t_per_mode.append(np.sum(self._T, axis = axis))
            self._tot_t_other.append(np.sum(self._T, axis = j))
            
        # total (T)
        self._tot = np.sum(self._T, dtype = 'int64')

        #.astype(np.float64)
        
        # total^2 (T^2)
        self._square_tot = np.power(self._tot, 2)


        # 2 / total (2/T)
        self._two_divided_by_tot = 2 / self._tot
  

    def _update_intermediate_values_after_move(self, dimension):
        """
        Computes Omegas and Gamma

        """
        

        d = (dimension + 1) % self._n_modes

        
        # omegas
        if self.optimization_strategy in ['AVG', 'ALT', 'ALT2']:
            self._omega = np.zeros(self._n_modes)
            for i in range(self._n_modes):
                numerator = np.nansum(np.power(self._tot_t_per_mode[i],2))
                self._omega[i] = 1 - np.true_divide(numerator, self._square_tot)
        else:
           self._omega = np.zeros(2)
           numerator_0 = np.nansum(np.power(self._tot_t_per_mode[d],2))
           self._omega[0] = 1 - np.true_divide(numerator_0, self._square_tot)
           numerator_1 = np.nansum(np.power(self._tot_t_other[d], 2))
           self._omega[1] = 1 - (numerator_1 / self._square_tot)

        # gamma
        numerator_gamma = np.nansum(np.true_divide(np.sum(np.power(self._T, 2), axis = d), self._tot_t_other[d]))
        self._gamma = 1 - np.true_divide(numerator_gamma, self._tot)


        



    def _perform_move(self, dimension):
        """
        Perform a single move to improve the partition on rows.

        :param dimension: the mode we are considering (x = 0, y = 1, z = 2)
        :return:
        """

        logging.debug("[INFO] Special Case: {0}".format(self._special_case))
        if self._special_case >= 0:
            self._actual_item[dimension] += 1
            selected_element = (self._actual_item[dimension]) % self._n[dimension]
            selected_source_cluster = self._assignment[dimension][selected_element]
            lambdas, sum_lambdas = self._compute_lambdas(selected_element, dimension)
            #logging.debug("[INFO] selected_source_cluster: {0}".format(selected_source_cluster))
            #logging.debug("[INFO] number of elements in the cluster: {0}".format(np.shape(np.where(self._assignment[dimension] == selected_source_cluster)[0])))

        else:
            # select a random cluster on the considered mode
            selected_source_cluster = randint(1, self._n_clusters[dimension] * 100) % self._n_clusters[dimension]
            logging.debug("[INFO] selected_source_cluster: {0}".format(selected_source_cluster))
            logging.debug("[INFO] number of elements in the cluster: {0}".format(np.shape(np.where(self._assignment[dimension] == selected_source_cluster)[0])))

            # select a random element of selected_cluster
            if np.shape(np.where(self._assignment[dimension] == selected_source_cluster))[1] != 0:
                selected_element = choice(np.where(self._assignment[dimension]== selected_source_cluster)[0])
                lambdas, sum_lambdas = self._compute_lambdas(selected_element, dimension)
            #print(selected_element)
                logging.debug("[INFO] selected_element: {0}".format(selected_element))

        
        #logging.debug("[INFO] lambdas: {0}".format(lambdas))
        #logging.debug("[INFO] sum_lambdas: {0}".format(sum_lambdas))

        if self.optimization_strategy in ['AVG', 'ALT', 'ALT2']:
            computed_tau, delta_tau_0 = self._delta_tau(sum_lambdas, lambdas, selected_source_cluster, dimension)
        else:
            computed_tau, delta_tau_0 = self._delta_tau_agg(sum_lambdas, lambdas, selected_source_cluster, dimension)
        e_min = self._choose_cluster(computed_tau, delta_tau_0, selected_source_cluster)

        if e_min == selected_source_cluster:
            assignment = self._n_clusters[dimension]
        else:
            assignment = e_min

        go_on_normally = True
        if self._n_clusters[dimension] == self.min_n_clusters:
            # if the number of row cluster is already equal to min check that the move will not delete a cluster
            go_on_normally = self._check_clustering_size(selected_source_cluster, dimension)
            logging.debug("[INFO] go_on_normally: {0}".format(go_on_normally))

        if go_on_normally and e_min >= 0:
            # go with the move
            self._assignment[dimension][selected_element] = assignment
            self._modify_cluster(lambdas, selected_source_cluster, e_min, dimension)
            self._iterations_without_moves = 0
            
			
        else:
            logging.debug("[INFO] Ignored move of row {2} from row cluster {0} to {1}".format(selected_source_cluster,
                                                                                             e_min, selected_element))
            self._iterations_without_moves += 1
        self._update_intermediate_values_after_move(dimension)



        
    def _compute_lambdas(self, selected_element, dimension):
        """
        Compute lambda values related to the selected element.

        In particular:
        * lambdas, matrix, shape = (#clusters on the first other mode, #clusters on the second other mode)
                    contains, for each couple of clusters on the two other modes, the sum of data related to the selected element;
        * sum_lambdas, float
                    contains the sum of lambdas

        :param selected_element: int, the id of the selected element
        :param dimension: the mode we are considering (x = 0, y = 1, z = 2)
        :return: a pair (lambdas, sum_lambdas),
                    see the method description for more details
        """

        lambdas = self._sum_data_per_clusters(selected_element, dimension)
        sum_lambdas = np.sum(lambdas)

        return lambdas, sum_lambdas

    def _sum_data_per_clusters(self, index, dimension):   # <---- DA RIVEDERE!!!!!
        """
        Fixed an element, computes for each couple of clusters on the two other modes,
        the sum of data related to the selected element;

        :param index: int, the index of the element to consider
        :param dimension: the mode we are considering (x = 0, y = 1, z = 2)
        :return:
        """

        other_dimensions = [i for i in range(self._n_modes) if i != dimension]
        temp_T = np.take(self._dataset, index, axis = dimension)


        np.transpose(self._T, tuple([dimension]) + tuple([i for i in range(self._n_modes) if i != dimension]))


        for k in range(self._n_modes - 1):
            if k > 0:
                #temp_T = np.transpose(temp_T, np.transpose(tuple(range(1,self._n_modes -1)) + tuple([0])))
                temp_T = np.transpose(temp_T, tuple(range(1,self._n_modes -1)) + tuple([0]))

            for j in range(self._n_clusters[other_dimensions[k]]):
                l = list(self._assignment[other_dimensions[k]] == j) + [False] * j
                T = temp_T[l]
                T = np.sum(T, axis = 0)
                temp_T = np.concatenate((temp_T, T.reshape(tuple([1]) + np.shape(T))))

                
            temp_T = temp_T[-self._n_clusters[other_dimensions[k]]:]
            
        self.temp_T =  np.copy(temp_T)
        temp_T = np.transpose(temp_T, tuple(range(1,self._n_modes - 1)) + tuple([0]))
        
        return temp_T

    def _delta_tau(self, sum_lambdas, lambdas, original_cluster, dimension):
        """
        Compute the delta tau values. Fixed a source_cluster computes the delta tau values
        for each of the other existent cluster (plus the empty one)
      
        :param sum_lambdas: the sum of lambdas
        :param lambdas: matrix related to the element that should be moved
        :param original_cluster: the cluster currently containing the element
        :param dimension: int, the mode in wich perform the move (x = 0, y = 1, z = 2)
        :return: list of int, the tau value for each row cluster

        delta_tau_0 = (omega_dimension * 2/T * a + gamma * b *c)/(omega_dimension^2 - omega_dimension * b * c)
        delta_tau_j = 1/(omega_j * T) (e - d + k)
        """

        _T = np.transpose(self._T, tuple([dimension]) + tuple([i for i in range(self._n_modes) if i != dimension]))
        temp_T = np.copy(_T)
        temp_T[original_cluster] = 0
        temp_tot_t_per_main_mode = np.sum(temp_T, axis = tuple(range(1,self._n_modes)))
        temp_tot_t_other = []
        sum_other = []
        for j in range(self._n_modes):
            temp_tot_t_other.append(np.sum(temp_T, axis = j))
            sum_other.append(np.sum(_T, axis = j))    
        

        c = np.nan_to_num(np.true_divide(np.multiply(sum_lambdas, 2), self._square_tot)) # 2 lambda.. / T^2
        a_division = np.nan_to_num(np.true_divide(lambdas, self._tot_t_other[dimension])) #(lambda_jk / t.jk) tensor of dimension num_y_clust * num_z_clust * .... * num_{last_mode}_clusters
        a_subtraction = np.subtract(_T[original_cluster], lambdas) #(tb11, ..., tbml) - (lambda_11, ..., lambda_ml) tensor of dimension num_y_clust * num_z_clust * .... * num_{last_mode}_clusters

        a1 = np.subtract(a_subtraction, temp_T) #tensor num_x_clusters * num_y_clusters * num_z_clusters * .... * num_{last_mode}_clusters
        a = np.sum(np.multiply(a_division, a1), axis = tuple(range(1,self._n_modes))) # vector of dimension num_x_clusters
        b = temp_tot_t_per_main_mode - np.nansum(a_subtraction) # vector of dimension num_x_clusters

        denominator = self._omega[dimension]**2 -self._omega[dimension] * b * c

        delta_tau_0 = np.nan_to_num(np.true_divide(np.true_divide(2 * self._omega[dimension], self._tot) * a + (self._gamma * c) * b, denominator))
        
        computed_taus = np.copy(delta_tau_0)
        #print(f"delta_0: {delta_tau_0}")
        for j in range(1, self._n_modes):
            
            if j <= dimension:
                h = j-1
            else:
                h = j
            __T = np.transpose(_T, tuple([0,j]) + tuple([i for i in range(1,self._n_modes) if i != j]))
            __a_subtraction = np.transpose(a_subtraction, tuple([j-1]) + tuple([i for i in range(self._n_modes -1) if i != j-1]))
            #k = np.sum(np.subtract(np.nan_to_num(np.true_divide(np.power(_T[original_cluster], 2), sum_other[j][original_cluster])),
            #                         np.nan_to_num(np.true_divide(np.power(a_subtraction, 2), np.sum(a_subtraction, axis = j-1)))))#scalar      
            k = np.sum(np.subtract(np.nan_to_num(np.true_divide(np.power(__T[original_cluster], 2), sum_other[j][original_cluster])),
                                     np.nan_to_num(np.true_divide(np.power(__a_subtraction, 2), np.sum(__a_subtraction, axis = 0)))))#scalar      

        
            T1 = temp_T + lambdas #tensor num_x_clusters * num_y_clusters * num_z_clusters
            d1 = np.sum(np.power(T1, 2), axis = j) # tensor of dimension num_x_clusters * num_z_clusters
            d = np.sum(np.nan_to_num(np.true_divide(d1, temp_tot_t_other[j] + np.sum(lambdas, axis = j-1))), axis = tuple(range(1,self._n_modes - 1))) # vector of dimension num_x_clusters
            
        
            temp_tot_t_other[j][original_cluster] = 1
            e1 = np.sum(np.power(temp_T, 2), axis = j) # tensor of dimension num_x_clusters * num_z_clusters
            e = np.sum(np.nan_to_num(np.true_divide(e1, temp_tot_t_other[j])), tuple(range(1,self._n_modes - 1))) # vector of dimension num_x_clusters

            #print(f"delta_{j} : {np.true_divide(e - d + k,  self._omega[h] * self._tot)}")
            computed_taus += np.nan_to_num(np.true_divide(e - d + k,  self._omega[h] * self._tot))



        #logging.debug("[INFO] taus : {0}".format((delta_tau_0, delta_tau_1, delta_tau_2)))


        # check if the source is a singleton cluster and force useless move to empty cluster to 0.0
        is_singleton = not self._check_clustering_size(original_cluster, dimension)
        if is_singleton:
            computed_taus[original_cluster] = 0.0


        if self.optimization_strategy == 'ALT2':
            d0 = delta_tau_0 < 0
            d1 = delta_tau_0 == 0
            d2 = computed_taus < 0
            d = d0 + (d1 * d2) # clusters with delta_tau_0 ==0 and computed_taus < 0, or delta_tau_0 negative
            computed_taus = computed_taus * d

        return computed_taus, delta_tau_0

    def _delta_tau_agg(self, sum_lambdas, lambdas, original_cluster, dimension):
        """
        Compute the delta tau values for row clusters. Fixed a source_cluster computes the delta tau values
        for each of the other existent cluster (plus the empty one)

        :param tot_t_per_cc: the sum of t values grouped by column cluster
        :param sum_lambdas: the sum of lambdas per view
        :param lambdas: for each view and column cluster the difference due to the move of the element
        :param original_cluster: the cluster currently containing the element
        :return: list of int, the tau value for each row cluster
        """

        """# VECCHIO METODO 
        _T = np.transpose(self._T, tuple([dimension]) + tuple([i for i in range(self._n_modes) if i != dimension]))
        temp_T = np.copy(_T)
        temp_T[original_cluster] = 0
        temp_tot_t_per_main_mode = np.sum(temp_T, axis = tuple(range(1,self._n_modes)))
        #temp_tot_t_other = []
        #sum_other = []
        #for j in range(self._n_modes):
            #temp_tot_t_other.append(np.sum(temp_T, axis = j))
            #sum_other.append(np.sum(_T, axis = j))


        c = np.nan_to_num(np.true_divide(np.multiply(sum_lambdas, 2), self._square_tot)) # 2 lambda.. / T^2
        a_division = np.nan_to_num(np.true_divide(lambdas, self._tot_t_other[dimension])) #(lambda_jk / t.jk) tensor of dimension num_y_clust * num_z_clust * .... * num_{last_mode}_clusters
        a_subtraction = np.subtract(_T[original_cluster], lambdas) #(tb11, ..., tbml) - (lambda_11, ..., lambda_ml) tensor of dimension num_y_clust * num_z_clust * .... * num_{last_mode}_clusters

        a1 = np.subtract(a_subtraction, temp_T) #tensor num_x_clusters * num_y_clusters * num_z_clusters * .... * num_{last_mode}_clusters
        a = np.sum(np.multiply(a_division, a1), axis = tuple(range(1,self._n_modes))) # vector of dimension num_x_clusters
        b = temp_tot_t_per_main_mode - np.nansum(a_subtraction) # vector of dimension num_x_clusters

        denominator = self._omega[0]**2 -self._omega[0] * b * c

        delta_tau_0 = np.nan_to_num(np.true_divide(np.true_divide(2 * self._omega[0], self._tot) * a + (self._gamma * c) * b, denominator))



##        k = np.sum(np.subtract(np.nan_to_num(np.true_divide(np.power(_T[original_cluster], 2), temp_tot_t_per_main_mode[original_cluster])),
##                                     np.nan_to_num(np.true_divide(np.power(a_subtraction, 2), temp_tot_t_per_main_mode[original_cluster] + sum_lambdas))))#scalar      
        k = np.sum(np.subtract(np.nan_to_num(np.true_divide(np.power(_T[original_cluster], 2), self._tot_t_per_mode[dimension][original_cluster])),
                                     np.nan_to_num(np.true_divide(np.power(a_subtraction, 2), self._tot_t_per_mode[dimension][original_cluster] + sum_lambdas))))#scalar      


        T1 = temp_T + lambdas #tensor num_x_clusters * num_y_clusters * num_z_clusters
        d1 = np.sum(np.power(T1, 2), axis = tuple(range(1, self._n_modes))) # vector of dimension num_x_clusters 
        d = np.nan_to_num(np.true_divide(d1, temp_tot_t_per_main_mode + sum_lambdas)) # vector of dimension num_x_clusters
        e1 = np.sum(np.power(temp_T, 2), axis = tuple(range(1, self._n_modes))) # tensor of dimension num_x_clusters
        e = np.nan_to_num(np.true_divide(e1, temp_tot_t_per_main_mode)) # vector of dimension num_x_clusters


        


        delta_tau_1 = np.nan_to_num(np.true_divide(e - d + k,  self._omega[1] * self._tot))



        computed_taus = np.add(delta_tau_0, delta_tau_1)

        
        # check if the source is a singleton cluster and force useless move to empty cluster to 0.0
        is_singleton = not self._check_clustering_size(original_cluster, dimension)
        if is_singleton:
            computed_taus[original_cluster] = 0.0
            delta_tau_0[original_cluster] = 0.0

        if self.optimization_strategy == 'AGG2':

            d0 = delta_tau_0 < 0
            d1 = delta_tau_0 == 0
            d2 = computed_taus < 0
            d = d0 + (d1 * d2) # clusters with delta_tau_0 == 0 and computed_taus < 0, or delta_tau_0 negative
            computed_taus = computed_taus * d


        """

        #NUOVO METODO

        _T = np.copy(np.transpose(self._T, tuple([dimension]) + tuple([i for i in range(self._n_modes) if i != dimension])))

        c = np.nan_to_num(np.true_divide(np.multiply(sum_lambdas, 2), self._square_tot)) # 2 lambda.. / T^2
        a_division = np.nan_to_num(np.true_divide(lambdas, self._tot_t_other[dimension])) #(lambda_jk / t.jk) tensor of dimension num_y_clust * num_z_clust * .... * num_{last_mode}_clusters
        a_subtraction = np.subtract(_T[original_cluster], lambdas) #(tb11, ..., tbml) - (lambda_11, ..., lambda_ml) tensor of dimension num_y_clust * num_z_clust * .... * num_{last_mode}_clusters
        k = np.sum(np.subtract(np.nan_to_num(np.true_divide(np.power(_T[original_cluster], 2), self._tot_t_per_mode[dimension][original_cluster])),
                                     np.nan_to_num(np.true_divide(np.power(a_subtraction, 2), self._tot_t_per_mode[dimension][original_cluster] + sum_lambdas))))#scalar      





        

        _T[original_cluster] = 0
        temp_tot_t_per_main_mode = np.sum(_T, axis = tuple(range(1,self._n_modes)))



        
        a1 = np.subtract(a_subtraction, _T) #tensor num_x_clusters * num_y_clusters * num_z_clusters * .... * num_{last_mode}_clusters
        a = np.sum(np.multiply(a_division, a1), axis = tuple(range(1,self._n_modes))) # vector of dimension num_x_clusters
        b = temp_tot_t_per_main_mode - np.nansum(a_subtraction) # vector of dimension num_x_clusters

        denominator = self._omega[0]**2 -self._omega[0] * b * c

        delta_tau_0 = np.nan_to_num(np.true_divide(np.true_divide(2 * self._omega[0], self._tot) * a + (self._gamma * c) * b, denominator))
        del(a_division, a_subtraction,c, a1, a,b)



##        k = np.sum(np.subtract(np.nan_to_num(np.true_divide(np.power(_T[original_cluster], 2), temp_tot_t_per_main_mode[original_cluster])),
##                                     np.nan_to_num(np.true_divide(np.power(a_subtraction, 2), temp_tot_t_per_main_mode[original_cluster] + sum_lambdas))))#scalar      


        #T1 = temp_T + lambdas #tensor num_x_clusters * num_y_clusters * num_z_clusters
        d1 = np.sum(np.power(_T+lambdas, 2), axis = tuple(range(1, self._n_modes))) # vector of dimension num_x_clusters 
        d = np.nan_to_num(np.true_divide(d1, temp_tot_t_per_main_mode + sum_lambdas)) # vector of dimension num_x_clusters
        e1 = np.sum(np.power(_T, 2), axis = tuple(range(1, self._n_modes))) # tensor of dimension num_x_clusters
        e = np.nan_to_num(np.true_divide(e1, temp_tot_t_per_main_mode)) # vector of dimension num_x_clusters


        


        delta_tau_1 = np.nan_to_num(np.true_divide(e - d + k,  self._omega[1] * self._tot))



        computed_taus = np.add(delta_tau_0, delta_tau_1)
        del(d1,d,e1,e,delta_tau_1)

        
        # check if the source is a singleton cluster and force useless move to empty cluster to 0.0
        is_singleton = not self._check_clustering_size(original_cluster, dimension)
        if is_singleton:
            computed_taus[original_cluster] = 0.0
            delta_tau_0[original_cluster] = 0.0

        if self.optimization_strategy == 'AGG2':

            d0 = delta_tau_0 < 0
            d1 = delta_tau_0 == 0
            d2 = computed_taus < 0
            d = d0 + (d1 * d2) # clusters with delta_tau_0 == 0 and computed_taus < 0, or delta_tau_0 negative
            computed_taus = computed_taus * d
       

        return computed_taus, delta_tau_0


    def _choose_cluster(self, computed_tau, delta_tau_0, selected_source_cluster):  
        """
        It chooses the cluster where to move the selected element
        """
        e_min = -1
        if self.optimization_strategy in ['AVG','AGG']:
            min_delta_tau = np.min(computed_tau)
            if min_delta_tau < 0:
                equal_solutions = np.where(min_delta_tau == computed_tau)[0]
                if len(equal_solutions) > 1:
                    min_tau_0 = np.min(delta_tau_0[equal_solutions])
                    e_min = np.where(min_tau_0 == delta_tau_0)[0][0]
                else:
                    e_min = equal_solutions[0]
            elif min_delta_tau == 0:
                equal_solutions = np.where(min_delta_tau == computed_tau)[0]
                min_tau_0 = np.min(delta_tau_0[equal_solutions])
                if min_tau_0 < 0:
                    if len(equal_solutions) > 1:
                        e_min = np.where(min_tau_0 == delta_tau_0)[0][0]
                    else:
                        e_min = equal_solutions[0]
                

                
        elif self.optimization_strategy in ['ALT2','AGG2']:
            if np.all(computed_tau == 0):
                e_min = -1
            else:
                min_delta_tau = np.min(computed_tau[computed_tau != 0])
                equal_solutions = np.where(min_delta_tau == computed_tau)[0]
                if len(equal_solutions) > 1:
                    min_tau_0 = np.min(delta_tau_0[equal_solutions])
                    e_min = np.where(min_tau_0 == delta_tau_0)[0][0]
                else:
                    e_min = equal_solutions[0]

        else:
            min_delta_0 = np.min(delta_tau_0)
            if min_delta_0 < 0:
                equal_solutions = np.where(min_delta_0 == delta_tau_0)[0]
                if len(equal_solutions) > 1:
                    min_delta_tau = np.min(computed_tau[equal_solutions])
                    e_min = np.where(min_delta_tau == computed_tau)[0][0]
                else:
                    e_min = equal_solutions[0]
            elif min_delta_0 == 0:
                equal_solutions = np.where(min_delta_0 == delta_tau_0)[0]
                min_delta_tau = np.min(computed_tau[equal_solutions])
                if min_delta_tau < 0:
                    if len(equal_solutions) > 1:
                        e_min = np.where(min_delta_tau == computed_tau)[0][0]
                    else:
                        e_min = equal_solutions[0]

        return e_min
        
    def _update_T_derived_fields(self):
        """
        Initialize the values that will be used in the computation of Delta tau

        :return:
        """

        logging.debug("[INFO] Update fields derived by the contingency matrix...")


        # tot_t_per_mode[j]: Vector (t1.., ..., tm..)
        # tot_t_other[j]: tensor of n-1 modes
        self._tot_t_per_mode = []
        self._tot_t_other = []

        for j in range(self._n_modes):
            axis = tuple([i for i in range(self._n_modes) if i != j])
            self._tot_t_per_mode.append(np.sum(self._T, axis = axis))
            self._tot_t_other.append(np.sum(self._T, axis = j))
    
    
    def _modify_cluster(self, lambda_t, source_c, destination_c, dimension):
        """
        It calls one of the following functions, according to the mode considered:

        - self._modify_x_cluster
        - self._modify_y_cluster
        - self._modify_z_cluster

        :param lambda_t: matrix, values of the element we want to move
        :param source_c: int, the id of the original cluster
        :param destination_c: int, the id of the destination cluster
        :param dimension: int, the mode in wich perform the move (x = 0, y = 1, z = 2)
        :return:
        """

        
        lambda_tot = np.sum(lambda_t)

        
        
        if destination_c == source_c:
            # case 1) the destination cluster is a new one
            logging.debug("[INFO] Create new cluster {0}".format(destination_c))

            # add one dimension on the first mode for the new cluster
            new_shape = [1] *self._n_modes
            for j in range(self._n_modes):
                if j <dimension:
                    new_shape[j] = lambda_t.shape[j]
                elif j > dimension:
                    new_shape[j] = lambda_t.shape[j-1]

            self._T = np.concatenate((self._T, np.reshape(lambda_t, new_shape)), axis = dimension)

            # update the source    
            b = "self._T[" + ":," * dimension + "source_c] -= lambda_t"
            exec(b)

            #######################      <---- Verificare se serve

            #self._tot_t_per_mode[dimension][source_c] -= lambda_t            
            #self._tot_t_per_mode[dimension] = np.concatenate((self._tot_t_per_mode[dimension],[lambda_tot]))

            self._n_clusters[dimension] += 1

        else:
            # case 2) the destination cluster already exists
            # we move the object x from the original cluster to the destination cluster
            logging.debug("[INFO] Move element from cluster {0} to {1}".format(source_c, destination_c))

            # Update Contingency table _T
            b = "self._T[" + ":," * dimension + "source_c] -= lambda_t"
            c = "self._T[" + ":," * dimension + "destination_c] += lambda_t"
            exec(b)
            exec(c)



            #self._tot_t_per_x[source_c] -= lambda_tot
            #self._tot_t_per_x[destination_c] += lambda_tot
            #self._tot_t_per_xy[source_c] -= lambda_y
            #self._tot_t_per_xz[source_c] -= lambda_z
            #self._tot_t_per_xy[destination_c] += lambda_y
            #self._tot_t_per_xz[destination_c] += lambda_z


            # check that the original cluster has at least one remaining element
            is_empty = not self._check_clustering_size(source_c,dimension , min_number_of_elements=1)   # <---- VERIFICARE!

            if is_empty:
                # compact the contingency matrix
                # delete the source cluster
                self._T = np.delete(self._T, source_c, dimension)
                
                # update the total values removing the source cluster item
                #self._tot_t_per_x = np.delete(self._tot_t_per_x, source_c)
                #self._tot_t_per_xy = np.delete(self._tot_t_per_xy, source_c, axis = 0)
                #self._tot_t_per_xz = np.delete(self._tot_t_per_xz, source_c, axis = 0)

                # update the assignments to reflect the new cluster ids
                for di in range(self._n[dimension]):
                    if self._assignment[dimension][di] > source_c:
                        self._assignment[dimension][di] -= 1

                self._n_clusters[dimension] -= 1



        self._update_T_derived_fields()
        self._performed_moves[dimension] += 1
        

    def _check_clustering_size(self, cluster_id, dimension, min_number_of_elements=2):
        """
        Check if the specified cluster has at least min_number_of_elements elements.
        Returns True if the cluster contains at least the specified number of elements, False otherwise.

        :param cluster_id: int, the id of the cluster that contains the element at this moment
        :param dimension: the mode we are considering (x = 0, y = 1, z = 2)
        :param min_number_of_elements: int, default 2, the min number of elements that the cluster should have
        :return: boolean, True if the cluster has at least min_number_of_elements elements, False otherwise
        """

        for rc in self._assignment[dimension]:
            if rc == cluster_id:
                min_number_of_elements -= 1
            if min_number_of_elements <= 0:
                # stop when the min number is found
                return True

        return False

    def _compute_taus(self):
        """
        Compute the value of tau_x, tau_y and tau_z

        :return: a tuple (tau_x, tau_y, tau_z)
        """
        tau = np.zeros(self._n_modes)
        for j in range(self._n_modes):
            a = np.sum(np.nan_to_num(np.true_divide(np.sum(np.power(self._T, 2), axis = j), self._tot_t_other[j]))) # scalar
            b = np.true_divide(np.sum(np.power(self._tot_t_per_mode[j], 2)), self._square_tot) #scalar
            tau[j] = np.nan_to_num(np.true_divide(np.true_divide(a, self._tot) - b, 1 - b))

        
        #logging.debug("[INFO] a_x, a_y, a_z, b_x, b_y, b_z: {0},{1}, {2}, {3}, {4}, {5}".format(a_x, a_y, a_z, b_x, b_y, b_z))

        return tau


if __name__ == '__main__':
    from CreateMatrix import CreateTensor
    import logging
    logger = logging.getLogger()
    logger.setLevel(logging.ERROR)
    T1,sol = CreateTensor([100,99,98],[5,4,3],.2)
    V = np.array([[[[0,1,0,2],[1,0,1,1],[1,1,1,0]],[[1,1,1,1],[0,0,1,0],[5,1,1,3]]],[[[2,1,0,2],[2,0,1,1],[2,1,1,3]],[[2,1,1,1],[2,0,1,0],[0,1,1,0]]]])
    #F, sol = CreateTensor([100,100,20,20,20],[5,3,3,3,2],.3)
    model = CoClust(3000,optimization_strategy = 'AGG2')
    
