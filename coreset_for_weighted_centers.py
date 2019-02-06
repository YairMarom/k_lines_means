#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################


from __future__ import division
import copy
import numpy as np
from set_of_points import SetOfPoints
import math



class CoresetForWeightedCenters:
    """
    A class that includes all the main API of the weighted centers coreset
    """

    def __init__(self, parameters_config):
        self.parameters_config = parameters_config

    ######################################################################

    def recursive_robust_median(self, P, k, median_sample_size, recursive_median_closest_to_median_rate):
        """
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            median_closest_rate (float) : the size of closest points to the median relative to the set
            recursive_median_closest_to_median_rate (float) : parameter for the median

        Returns:
            [np.ndarray, SetOfPoints]: the recursive robust median of P and its closest points. See Alg. 1 in the paper;
        """

        assert k > 0, "k is not a positive integer"
        assert recursive_median_closest_to_median_rate < 1 and recursive_median_closest_to_median_rate > 0, "closest_rate2 not in (0,1)"
        assert P.get_size() != 0, "Q size is zero"

        minimum_number_of_points_in_iteration = int(math.log(P.get_size())) #for stop condition
        Q = copy.deepcopy(P)
        q = []
        for i in range(k):
            size_of_sample = median_sample_size
            q = Q.get_sample_of_points(size_of_sample)
            Q = Q.get_closest_points_to_set_of_points(q, recursive_median_closest_to_median_rate, type="by rate") #the median closest points

            size = Q.get_size()
            if size <= minimum_number_of_points_in_iteration:
                break
        return [q, Q]

    ######################################################################

    def coreset(self, P, k, m):
        """
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            median_sample_size (float) : parameter for the recursive median
            closest_to_median_rate (float) : parameter for the recursive median

        Returns:
            SetOfPoints: the coreset of P for k weighted centers. See Alg. 2 in the paper;
        """
        median_sample_size = self.parameters_config.median_sample_size
        closest_to_median_rate = self.parameters_config.closest_to_median_rate
        assert k > 0, "k is not a positive integer"
        assert m > 0, "m is not a positive integer"
        assert P.get_size() != 0, "Q size is zero"
        #number_of_remains_multiply_factor = self.parameters_config.number_of_remains_multiply_factor
        max_sensitivity_multiply_factor = self.parameters_config.max_sensitivity_multiply_factor
        minimum_number_of_points_in_iteration = self.parameters_config.number_of_remains
        Q = copy.deepcopy(P)
        temp_set = SetOfPoints()
        max_sensitivity = -1
        while True:
            [q_k, Q_k] = self.recursive_robust_median(Q, k, self.parameters_config.median_sample_size, self.parameters_config.closest_to_median_rate) #get the recursive median q_k and its closest points Q_k
            if Q_k.get_size() == 0:
                break
            Q_k.set_sensitivities(k) # sets all the sensitivities in Q_k as described in line 5 in main alg.
            current_sensitivity = Q_k.get_arbitrary_sensitivity()
            if current_sensitivity > max_sensitivity:
                max_sensitivity = current_sensitivity #we save the maximum sensitivity in order to give the highest sensitivity to the points that remains in Q after this loop ends
            temp_set.add_set_of_points(Q_k) #since we remove Q_k from Q each time, we still wan to save every thing in order to go over the entire points after this loop ends and select from them and etc., so we save everything in temp_set
            Q.remove_from_set(Q_k)
            size = Q.get_size()
            Q_k_weigted_size = Q_k.get_sum_of_weights()
            if size <= minimum_number_of_points_in_iteration or Q_k_weigted_size == 0: # stop conditions
                break
        if Q.get_size() > 0:
            Q.set_all_sensitivities(max_sensitivity * max_sensitivity_multiply_factor) # here we set the sensitivities of the points who left to the highest - since they are outliers with a very high probability
            temp_set.add_set_of_points(Q) #and now temp_set is all the points we began woth - just with updated sensitivities
        T = temp_set.get_sum_of_sensitivities()
        temp_set.set_weights(T, m) #sets the weights as described in line 10 in main alg
        size = temp_set.get_size()
        probs = temp_set.get_probabilites().reshape(1,-1)[0] #probs is an array of n elements, where the i-th element is the probabilty to choose the ith point
        all_indices = np.asarray(range(size))
        indices_sample = np.unique(np.random.choice(all_indices, m, True, probs)) #pick a sample from the indices by the distribution defined by the sensitivities, as described in line 9 in main alg.
        #indices_sample = np.random.choice(all_indices, m, False, probs) #pick a sample from the indices by the distribution defined by the sensitivities, as described in line 9 in main alg.
        A = temp_set.points[indices_sample] # pick the points by the indices we sampled
        v = temp_set.weights[indices_sample].reshape(1,-1)[0] # pick the weights by the indices we sampled
        return SetOfPoints(A, v)

    ######################################################################

    def coreset_return_sensitivities(self, P, k, m):
        """
        Args:
            P (SetOfPoints) : set of weighted points
            k (int) : number of weighted centers
            median_sample_size (float) : parameter for the recursive median
            closest_to_median_rate (float) : parameter for the recursive median

        Returns:
            SetOfPoints: the coreset of P for k weighted centers. See Alg. 2 in the paper;
        """
        median_sample_size = self.parameters_config.median_sample_size
        assert k > 0, "k is not a positive integer"
        assert m > 0, "m is not a positive integer"
        assert P.get_size() != 0, "Q size is zero"
        number_of_remains_multiply_factor = self.parameters_config.number_of_remains_multiply_factor
        max_sensitivity_multiply_factor = self.parameters_config.max_sensitivity_multiply_factor
        minimum_number_of_points_in_iteration = k*number_of_remains_multiply_factor #int(math.log(P.get_size()))
        Q = copy.deepcopy(P)
        temp_set = SetOfPoints()
        max_sensitivity = -1
        flag1 = False
        flag2 = False
        while True:
            [q_k, Q_k] = self.recursive_robust_median(Q, k, self.parameters_config.median_sample_size, self.parameters_config.closest_to_median_rate) #get the recursive median q_k and its closest points Q_k
            if Q_k.get_size() == 0:
                flag1 = True
                continue
            Q_k.set_sensitivities(k) # sets all the sensitivities in Q_k as described in line 5 in main alg.
            current_sensitivity = Q_k.get_arbitrary_sensitivity()
            if current_sensitivity > max_sensitivity:
                max_sensitivity = current_sensitivity #we save the maximum sensitivity in order to give the highest sensitivity to the points that remains in Q after this loop ends
            temp_set.add_set_of_points(Q_k) #since we remove Q_k from Q each time, we still wan to save every thing in order to go over the entire points after this loop ends and select from them and etc., so we save everything in temp_set
            Q.remove_from_set(Q_k)
            size = Q.get_size()
            Q_k_weigted_size = Q_k.get_sum_of_weights()
            if size <= minimum_number_of_points_in_iteration or Q_k_weigted_size == 0: # stop conditions
                flag2 = True
                break
        if Q.get_size() > 0:
            Q.set_all_sensitivities(max_sensitivity * max_sensitivity_multiply_factor) # here we set the sensitivities of the points who left to the highest - since they are outliers with a very high probability
            temp_set.add_set_of_points(Q) #and now temp_set is all the points we began woth - just with updated sensitivities
        if len(temp_set.sensitivities) != len(P.points):
            x=2
        T = temp_set.get_sum_of_sensitivities()
        temp_set.set_weights(T, m) #sets the weights as described in line 10 in main alg
        temp_set.sort_by_indexes()
        return temp_set.sensitivities, temp_set.weights

