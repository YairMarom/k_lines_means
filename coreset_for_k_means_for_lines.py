#################################################################
#     Corset for k means for lines                              #
#     Paper: TBD                                                #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################

from __future__ import division
import copy
import numpy as np
from coreset_for_weighted_centers import CoresetForWeightedCenters
from set_of_lines import SetOfLines
from set_of_points import SetOfPoints


class CorsetForKMeansForLines:
    """
    A class that includes all the main API of the coreset for k-means for lines
    """

    def __init__(self, parameters_config):
        self.parameters_config = parameters_config

    ######################################################################

    def alpha_beta_approximation(self, L, k, sample_size):
        """
        This method gets a set L of n lines, number of required centers k>0, and returns a set P consist of k*log(n)
        points, that its sum of squared distances to L is an approximation of the sum of squared distances
        from the optimal k centers, up to a constant factor. See Alg. 2 in the paper.
        Args:
            L (SetOfLines) : a set of lines
            k (int) : number of centers
            sample_size (int) : the number of lines will be sampled at each iteration

        Returns:
            cost_from_L_to_B, a_b_approx_centers (float, np.ndarray): the cost from the k*log(n) centers that defined above,
                and the k*log(n) centers that approximate the cost to the optimal k centers up to a constant factor
        """
        size = L.get_size()
        assert size > 0, "L is empty"
        assert k > 0, "k <= 0"

        B = SetOfPoints()
        Q = copy.deepcopy(L)
        minimum_number_of_lines = self.parameters_config.a_b_approx_minimum_number_of_lines
        closest_to_centers_rate_in_a_b_approx = self.parameters_config.farthest_to_centers_rate_in_a_b_approx

        while size > minimum_number_of_lines and size > sample_size:
            sample_of_lines = Q.get_sample_of_lines(sample_size)
            G = sample_of_lines.get_all_intersection_points()
            if len(G) > 0:
                G = SetOfPoints(G)
                B.add_set_of_points(G)
                Q = Q.get_farthest_lines_to_centers(G, int(size*closest_to_centers_rate_in_a_b_approx), "by number")
            size = Q.get_size()
        if size > 0:
            G = Q.get_all_intersection_points()
            if len(G) > 0:
                G = SetOfPoints(G)
                B.add_set_of_points(G)

        return B

    ######################################################################

    def coreset(self, L, k, m):
        """
        This is the main function - that gets a set of n lines, set of n corresponding weights, integer k for number of
        centers and an integer m for the output size, and returns m weighted lines which are (k-eps) coreset, as define
        in npaper. See Alg. 1 in paper.
        Args:
            L (SetOfLines) : a set of lines
            k (int) : number of centers

        Returns:
            SetOfLines: the (k-eps) coreset of L
        """

        size = L.get_size()
        assert size > 0, "L is empty"
        assert k > 0, "k <= 0"

        if m >= size:
            return L

        sample_size_for_a_b_approx = self.parameters_config.sample_size_for_a_b_approx
        B = self.alpha_beta_approximation(L=L, k=k,sample_size=sample_size_for_a_b_approx)
        cost_to_B = L.get_sum_of_distances_to_centers(B)
        #L.plot_lines(a_b_approx_centers)

        sensitivities_first_argument = L.get_sensitivities_first_argument_for_centers(B)
        sensitivity_first_argument_sum = np.sum(sensitivities_first_argument)
        sensitivity_first_argument_normalized = sensitivities_first_argument / sensitivity_first_argument_sum #the first argument of the total sensitivities, that is the distance of each line to its closest center in a_b_approx_centers divided by the total cost from a_b_approx_centers to L

        cluster_indexes = L.get_indices_clusters(B)
        B_clustered = B.get_points_from_indices(cluster_indexes)
        B_clustered_points = B_clustered.points
        Q = copy.deepcopy(L)
        Q.displacements = B_clustered_points
        points_from_intersaction_of_lines_and_spheres = Q.displacements + Q.spans
        corset_for_weighted_centers = CoresetForWeightedCenters(self.parameters_config)
        sensitivity_second_argument, weights_second_argument = corset_for_weighted_centers.coreset_return_sensitivities(P=SetOfPoints(points_from_intersaction_of_lines_and_spheres,Q.weights),k=2*k, m=m)

        sensitivity_second_argument_sum = np.sum(sensitivity_second_argument) #the second argument of the sensitivity is the sensitivity of the projected lines - each line in L was projected to its closest center in a_b_approx_centers, and their sensitivity is the sensitivity of the intersection points with the unitspheres that are centers in each on of the a_b_approx_centers, as defined in paper
        sensitivity_second_argument_normalized = sensitivity_second_argument / sensitivity_second_argument_sum

        if cost_to_B == 0:
            sensitivities = sensitivity_second_argument_normalized.reshape(-1) #according to line 5 in Alg. 1 in paper
        else:
            sensitivities = 0.2*sensitivity_first_argument_normalized.reshape(-1) + 0.8*sensitivity_second_argument_normalized.reshape(-1) #according to line 5 in Alg. 1 in paper

        T = np.sum(sensitivities)
        probs = sensitivities/T
        probs_inv = (1 / probs)
        w_div_m = L.weights / m
        u = np.multiply(probs_inv.reshape(-1,1), w_div_m.reshape(-1,1)).reshape(-1) #this promises us that in the expectancy we would get a weighted m lines that approximate the original set L up to epsilon addative error
        Q = copy.deepcopy(L)
        Q.weights = u
        all_indices = np.asarray(range(size))
        indices_sample = np.unique(np.random.choice(all_indices, m, True, probs))
        #indices_sample = np.random.choice(all_indices, m, False, probs)
        S = Q.get_lines_at_indices(indices_sample)
        L_sum_of_weights = np.sum(L.weights)
        S_sum_of_weights = np.sum(S.weights)
        S.multiply_weights_by_value(L_sum_of_weights / S_sum_of_weights)
        return S

    ######################################################################

    def builed_sensitivities_first_arg_from_dict(self, centers_to_closest_lines_dict, cost_from_L_to_B):
        """
        TODO: complete
        :param centers_to_closest_lines_dict:
        :param cost_from_L_to_B:
        :return:
        """
        L_ordered = SetOfLines()
        L_tag = SetOfLines()
        sensitivities_first_argument_total = np.asarray([])
        for centers, closest_lines in centers_to_closest_lines_dict.items():
            indices_cluster = closest_lines.get_indices_clusters(centers)
            centeres_clustered_points = centers.points[indices_cluster]
            closest_lines_spans = closest_lines.spans
            closest_lines_displacements = closest_lines.displacements
            closest_lines_weights = closest_lines.weights

            centers_by_cluster_indices_minus_displacements = centeres_clustered_points - closest_lines_displacements
            centers_by_cluster_indices_minus_displacements_squared_norms = np.sum(
                np.multiply(centers_by_cluster_indices_minus_displacements,
                            centers_by_cluster_indices_minus_displacements), axis=1)
            centers_mul_spans_squared_norms = np.sum(
                np.multiply(centers_by_cluster_indices_minus_displacements, closest_lines_spans) ** 2, axis=1)
            all_unweighted_distances = centers_by_cluster_indices_minus_displacements_squared_norms - centers_mul_spans_squared_norms
            all_distances = np.multiply(all_unweighted_distances.reshape(-1, 1), closest_lines_weights.reshape(-1, 1)).reshape(-1)
            sensitivities_first_argument_current = all_distances / cost_from_L_to_B
            sensitivities_first_argument_total = np.append(sensitivities_first_argument_total, sensitivities_first_argument_current, axis=0)
            L_ordered.add_set_of_lines(closest_lines)


            closest_lines_tag = copy.deepcopy(closest_lines)
            closest_lines_tag.displacements = centeres_clustered_points
            #closest_lines_tag.normalized_lines_representation()
            L_tag.add_set_of_lines(closest_lines_tag)

        return L_ordered, L_tag, sensitivities_first_argument_total
