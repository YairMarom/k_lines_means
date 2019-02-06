#################################################################
#     Corset for k means for lines                              #
#     Paper: TBD                                                #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################
import copy
import math

import numpy as np

from set_of_points import SetOfPoints


class SetOfLines:
    """
    Class that represent a set of weighted lines in any d>0 dimensional space
    Attributes:
        spans (ndarray) : The spaning vectors. The i-th element is the spanning vector of the i-th line in the set
        displacements (ndarray) : The displacements. The i-th element is the displacement vector of the i-th line in the set
        dim (integer): The space's dimension
    """

    ##################################################################################

    def __init__(self, spans=None, displacements=None, weights=None, sen=None, lines=None, is_points=False):

        if spans is None:
            spans = []
        if displacements is None:
            displacements = []
        if weights is None:
            weights = []
        if sen is None:
            sen = []
        if lines is None:
            lines = []

        if is_points:
            self.dim = 2
            self.spans = []
            self.displacements = []
            for line in lines:
                v1 = np.asarray([line[0], line[1]])
                v2 = np.asarray([line[2], line[3]])
                span = v1 - v2
                displacement = v1
                self.spans.append(span)
                self.displacements.append(displacement)
            self.spans = np.asarray(self.spans)
            self.displacements = np.asarray(self.displacements)
            # self.normalized_lines_representation()
            self.normalize_spans()
            self.weights = np.ones(len(lines)).reshape(-1)
            self.sensitivities = np.ones(len(lines))
        else:
            size = len(spans)
            if size == 0:  # there is no lines in the set we got
                self.spans = []
                self.displacements = []
                self.weights = []
                self.sensitivities = []
                self.dim = 0
                return
            [_, self.dim] = np.shape(spans)
            self.spans = spans
            self.normalize_spans()
            self.displacements = displacements
            self.weights = weights
            self.sensitivities = sen
            # self.normalized_lines_representation()

    def blockshaped(self, arr, nrows, ncols):
        """
        Return an array of shape (n, nrows, ncols) where
        n * nrows * ncols = arr.size

        If arr is a 2D array, the returned array should look like n subblocks with
        each subblock preserving the "physical" layout of arr.
        """
        h, w = arr.shape
        return (arr.reshape(h // nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    ##################################################################################

    def get_all_intersection_points_optimized(self):
        """
        this method returns n(n-1) points, where each n-1 points in the n-1 points on each line that are closest to the
        rest n-1 lines.

        Args:
            ~

        Returns:
            np.ndarray: all the "intersection" points
        """
        assert self.get_size() > 0, "set is empty"

        spans = self.spans
        displacements = self.displacements
        dim = self.dim
        size = self.get_size()

        spans_repeat_each_point = np.repeat(spans, size,
                                            axis=0)  # that is a repeat of the spans, each span[i] is being repeated size times
        identity = np.identity(dim)
        identity_repeat_rows_all = np.repeat(identity.reshape(1, -1), size, axis=0).reshape(-1, dim)
        I_final = np.repeat(identity_repeat_rows_all, size, axis=0).reshape(size * dim,
                                                                            size * dim)  # the final is an identity matrix that is duplicated in rows and cols in factor of size
        G_G_T_all_permutations = np.outer(spans,
                                          spans)  # this is a 2 dimensional matrix of blocks, where the (i,j)-th block is spans[i]*spans[j]^T
        I_minus_G_G_T_all_permutations = I_final - G_G_T_all_permutations
        I_minus_G_G_tag_blocks = self.blockshaped(I_minus_G_G_T_all_permutations, dim,
                                                  dim)  # it will take the big matrix that is built from many clocks and returns a stack of blocks matrices
        I_minus_G_G_T_s = I_minus_G_G_tag_blocks[0:len(
            I_minus_G_G_tag_blocks):size + 1]  # this is a 1 dimensional matrix of blocks, where the i-th block is spans[i]*spans[i]^T
        I_minus_G_G_T_s_concatenated = I_minus_G_G_T_s.reshape(1, -1).T.reshape(-1,
                                                                                dim).T  # that is a 1 dimensional block matrix, where the i-th element is the matrix I-G_iG_i^T
        I_minus_G_G_T_s_F = np.dot(spans, I_minus_G_G_T_s_concatenated)
        I_minus_G_G_T_s_F = I_minus_G_G_T_s_F.reshape(-1, 1).reshape(-1,
                                                                     dim)  # in this matrix, the i-th index is the dot product of spans[j] and the k-th (I-GG^T), for j=i/size and k=i%size
        I_minus_G_G_T_s_F_inv = np.linalg.pinv(I_minus_G_G_T_s_F.reshape(size ** 2, dim,
                                                                         1))  # this matrix dimension is $size^2 \times dim$, where the i-th element is the point on the line j=i/size that are closest to the line m=i%size. that means: I_minus_G_G_T_s_F_inv[1] = ((I-G_1G_1^T)F_1)^+, I_minus_G_G_T_s_F_inv[2] = ((I-G_1G_1^T)F_2)^+,...,I_minus_G_G_T_s_F_inv[i] = ((I-G_jG_j^T)F_m)^+,
        I_minus_G_G_T_s_F_inv = I_minus_G_G_T_s_F_inv.reshape(size ** 2, dim)
        displacements_repeat_each_point = np.repeat(displacements, size, axis=0).reshape(size ** 2, dim)
        displacements_repeat_all = np.repeat(displacements.reshape(1, -1), size, axis=0).reshape(size ** 2, dim)
        f_minus_g = displacements_repeat_all - displacements_repeat_each_point  # this is a matrix where the i-th element is the substraction of g_j-f_m, where j=i/size and m=i%size
        I_minus_G_G_T_s_dot_f_minus_g = np.dot(f_minus_g, I_minus_G_G_T_s_concatenated)
        I_minus_G_G_T_s_dot_f_minus_g = I_minus_G_G_T_s_dot_f_minus_g.reshape(-1, 1).reshape(size, -1,
                                                                                             dim)  # this matrix contains more than it needs to contain. The i-th element is (I-G_iG_i^T)(f_k-g_l), and we do not need the cases where i!=l, that is why we take the right subset in the folowing two rows
        inner_steps = np.arange(0, size ** 2, size + 1)
        I_minus_G_G_T_s_dot_f_minus_g_s = I_minus_G_G_T_s_dot_f_minus_g[:, inner_steps]
        I_minus_G_G_T_s_dot_f_minus_g_s = I_minus_G_G_T_s_dot_f_minus_g_s.reshape(-1,
                                                                                  dim)  # this matrix is the right matrix, where the i-th element is (I-G_iG_i^T)(f_j-g_i)
        final = np.multiply(I_minus_G_G_T_s_F_inv,
                            I_minus_G_G_T_s_dot_f_minus_g_s)  # each row in this matrix is ((I-G_iG_i^T)F_j)^{+}(I-G_iG_i^T)(f_j-g_i)
        final_x_stars = np.sum(final, axis=1)  # that yields the scalar the fits ti Fx-b in each line
        F_x_s = np.multiply(spans_repeat_each_point.T, final_x_stars)
        F_x_s_minus_b = F_x_s.T + displacements_repeat_each_point  # reconstruct points from all the x stars
        indices = np.arange(0, len(F_x_s_minus_b), size + 1)
        F_x_s_minus_b = np.delete(F_x_s_minus_b, indices,
                                  axis=0)  # removing all the unnecessary "closest point on the i-th line in the set to the i-th line in the set"
        return F_x_s_minus_b

    ##################################################################################

    def get_all_intersection_points(self):
        """
        this method returns n(n-1) points, where each n-1 points in the n-1 points on each line that are closest to the
        rest n-1 lines.

        Args:
            ~

        Returns:
            np.ndarray: all the "intersection" points
        """
        assert self.get_size() > 0, "set is empty"

        spans = self.spans
        displacements = self.displacements
        dim = self.dim
        size = self.get_size()

        t = range(size)
        indexes_repeat_all_but_one = np.array(
            [[x for i, x in enumerate(t) if i != j] for j, j in enumerate(t)]).reshape(-1)

        spans_rep_each = spans[
            indexes_repeat_all_but_one]  # repeat of the spans, each span[i] is being repeated size times in a sequance
        spans_rep_all = np.repeat(spans.reshape(1, -1), size - 1, axis=0).reshape(-1,
                                                                                  dim)  # repeat of the spans, all the spans block is repeated size-1 times
        disp_rep_each = displacements[
            indexes_repeat_all_but_one]  # repeat of the displacements, each span[i] is being repeated size times in a sequance
        disp_rep_all = np.repeat(displacements.reshape(1, -1), size - 1, axis=0).reshape(-1,
                                                                                         dim)  # repeat of the displacements, all the spans block is repeated size-1 times

        W0 = disp_rep_each - disp_rep_all
        a = np.sum(np.multiply(spans_rep_each, spans_rep_each), axis=1)
        b = np.sum(np.multiply(spans_rep_each, spans_rep_all), axis=1)
        c = np.sum(np.multiply(spans_rep_all, spans_rep_all), axis=1)
        d = np.sum(np.multiply(spans_rep_each, W0), axis=1)
        e = np.sum(np.multiply(spans_rep_all, W0), axis=1)
        be = np.multiply(b, e)
        cd = np.multiply(c, d)
        be_minus_cd = be - cd
        ac = np.multiply(a, c)
        b_squared = np.multiply(b, b)
        ac_minus_b_squared = ac - b_squared
        s_c = be_minus_cd / ac_minus_b_squared
        """
        for i in range(len(s_c)):
            if np.isnan(s_c[i]):
                s_c[i] = 0
        """
        s_c_repeated = np.repeat(s_c.reshape(-1, 1), dim, axis=1)
        G = disp_rep_each + np.multiply(s_c_repeated, spans_rep_each)

        b = np.where(np.isnan(G))
        c = np.where(np.isinf(G))
        G2 = np.delete(G, np.concatenate((b[0], c[0]), axis=0), axis=0).reshape(-1, dim)

        if len(G2) == 0:  # that means all the lines are parallel, take k random points from the displacements set
            return displacements;

        b2 = np.where(np.isnan(G2))
        c2 = np.where(np.isinf(G2))
        d2 = np.sum(b2)
        e2 = np.sum(c2)
        f2 = d2 + e2
        if f2 > 0:
            x = 2

        return G2

    ##################################################################################

    def get_4_approx_points_ex_search(self, k):

        """
        This method returns k points that minimizes the sum of squared distances to the lines in the set, up to factor
        of 4.

        Args:
            k (int) : the number of required centers.

        Returns:
            np.ndarray: a set of k points that minimizes the sum of squared distances to the lines in the set, up to
            a constant factor.
        """

        assert k > 0, "k <= 0"
        assert self.get_size() > 0, "set is empty"

        dim = self.dim
        size = self.get_size()
        displacements = self.displacements
        spans = self.spans
        weights = self.weights

        intersection_points_before_uniqe = self.get_all_intersection_points()
        number_of_intersection_points = np.shape(intersection_points_before_uniqe.reshape(-1, dim))[0]
        if number_of_intersection_points == 1 or number_of_intersection_points == 0:
            intersection_points = intersection_points_before_uniqe
        else:
            intersection_points = np.unique(intersection_points_before_uniqe,
                                            axis=0)  # that is n(n-1) points - the union of every n-1 points on each line in the set that are closest to the n-1 other lines
        intersection_points_size = len(intersection_points)
        all_indices = np.asarray(range(intersection_points_size)).reshape(1, -1)
        all_indices_repeat = np.repeat(all_indices, k, axis=0)
        after_mesh_grid = np.array(np.meshgrid(*all_indices_repeat)).T.reshape(-1, k)
        all_k_combinations_of_all_incdices = np.unique(after_mesh_grid, axis=0).reshape(-1)
        intersection_points_repeat_each_row = np.repeat(intersection_points, size, axis=0).reshape(-1,
                                                                                                   dim)  # each one of the points is repeated for size times, in order to calculate the distance from each point to the n lines
        displacements_repeat_all = np.repeat(displacements.reshape(1, -1), intersection_points_size, axis=0).reshape(-1,
                                                                                                                     dim)  # each displacement repeated for intersection_points_size_size times to match the number of points
        spans_repeat_all = np.repeat(spans.reshape(1, -1), intersection_points_size, axis=0).reshape(-1,
                                                                                                     dim)  # the same for the spans
        intersection_points_repeat_each_row_minus_displacements_repeat_all = intersection_points_repeat_each_row - displacements_repeat_all  # first part of distance function between points and lines
        intersection_points_size_minus_displacements_squared_norms = np.sum(
            np.multiply(intersection_points_repeat_each_row_minus_displacements_repeat_all,
                        intersection_points_repeat_each_row_minus_displacements_repeat_all), axis=1)
        np.sum(np.multiply(intersection_points_repeat_each_row_minus_displacements_repeat_all, spans_repeat_all) ** 2,
               axis=1)
        intersection_points_minus_displacements_dot_spans = np.multiply(
            intersection_points_repeat_each_row_minus_displacements_repeat_all, spans_repeat_all) ** 2
        intersection_points_minus_displacements_dot_spans_squared_norms = np.sum(
            intersection_points_minus_displacements_dot_spans, axis=1)
        all_unweighted_distances = intersection_points_size_minus_displacements_squared_norms - intersection_points_minus_displacements_dot_spans_squared_norms  # last part of distance calculatoin between points and lines
        weights_repeat_all = np.repeat(weights.reshape(-1, 1), intersection_points_size, axis=0)
        all_weighted_distances = np.multiply(all_unweighted_distances.reshape(-1, 1), weights_repeat_all.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, size)
        all_distances = all_distances.reshape(intersection_points_size,
                                              size)  # the i,j-th element is the distance between the i-th point in intersection_points_size and the j-th line in the set
        all_k_combinations_of_all_distances = all_distances[
            all_k_combinations_of_all_incdices]  # this is an array of size_of_intersection_points_size*size elements, where the i,j-element is the distance between the i-th point in intersection_points_size to the j-th line in the set
        all_k_combinations_of_all_distances_reshaped = all_k_combinations_of_all_distances.reshape(-1, k,
                                                                                                   size)  # reshaped for later calculation
        distances_for_each_combination_of_k_points = np.min(all_k_combinations_of_all_distances_reshaped[:, :],
                                                            axis=1)  # the i,j-th item is the distance from the i-th k-tuple from all points to the i-th line in the set
        sum_of_distances_from_k_tuples_to_lines = np.sum(distances_for_each_combination_of_k_points,
                                                         axis=1)  # the i-th element in this array is the sum of squared distanes between the i-th point in all_k_combinations_of_intersection_points_size_repeat_each_row to the lines in the set
        sum_of_distances_from_k_tuples_to_lines_min_index = np.argmin(sum_of_distances_from_k_tuples_to_lines)
        all_k_combinations_of_all_incdices_reshaped = all_k_combinations_of_all_incdices.reshape(-1, k)
        final_min_indices = all_k_combinations_of_all_incdices_reshaped[
            sum_of_distances_from_k_tuples_to_lines_min_index]
        P_4_approx = intersection_points[final_min_indices]
        P_4_approx = np.unique(P_4_approx, axis=0)
        P_4_approx = SetOfPoints(P_4_approx)
        return P_4_approx

    ##################################################################################

    def get_4_approx_points(self, k):

        """
        This method returns k points that minimizes the sum of squared distances to the lines in the set, up to factor
        of 4.

        Args:
            k (int) : the number of required centers.

        Returns:
            np.ndarray: a set of k points that minimizes the sum of squared distances to the lines in the set, up to
            a constant factor.
        """

        assert k > 0, "k <= 0"
        assert self.get_size() > 0, "set is empty"

        dim = self.dim
        size = self.get_size()
        displacements = self.displacements
        spans = self.spans
        weights = self.weights

        intersection_points_before_uniqe = self.get_all_intersection_points()
        intersection_points = np.unique(intersection_points_before_uniqe,
                                        axis=0)  # that is n(n-1) points - the union of every n-1 points on each line in the set that are closest to the n-1 other lines
        number_of_intersection_points = np.shape(intersection_points.reshape(-1, dim))[0]
        if number_of_intersection_points <= k:
            P_4_approx = intersection_points_before_uniqe
        else:
            all_indices = np.asarray(range(len(intersection_points)))
            indices_sample = np.random.choice(all_indices, k, False)
            P_4_approx = intersection_points[indices_sample]
        if len(P_4_approx) == 0:
            x = 2
        P_4_approx = SetOfPoints(P_4_approx)
        if P_4_approx.indexes == []:
            x = 2
        return P_4_approx

    ###################################################################################

    def get_size(self):
        """
        Args:
            ~

        Returns:
            int: number of lines in the set
        """

        return np.shape(self.spans)[0]

    ##################################################################################

    def get_size(self):
        """
        Args:
            ~

        Returns:
            int: number of lines in the set
        """

        return np.shape(self.spans)[0]

    ##################################################################################

    def get_sample_of_lines(self, size_of_sample):
        """
        Args:
            size_of_sample (int) : the sample's size

        Returns:
            SetOfLines: sample consist of size_of_sample lines from the uniform distribution over the set
        """

        assert self.get_size() > 0, "set is empty"
        assert size_of_sample > 0, "size_of_sample <= 0"

        size = self.get_size()
        if size_of_sample >= size:
            return self
        else:
            all_indices = np.asarray(range(size))
            sample_indices = np.random.choice(all_indices, size_of_sample, False).tolist()
            sample_spans = np.take(self.spans, sample_indices, axis=0, out=None, mode='raise')
            sample_displacements = np.take(self.displacements, sample_indices, axis=0, out=None, mode='raise')
            try:
                sample_weights = np.take(self.weights, sample_indices, axis=0, out=None, mode='raise')
            except Exception as e:
                x = 2
        return SetOfLines(sample_spans, sample_displacements, sample_weights)

    ##################################################################################

    def get_indices_clusters(self, centers):
        """
        This method gets a set of k centers (points), and returns a size-dimensional row vector of indices in the range
        [0,k-1], where every number num in the i-th item indicates that centers[i] is the center that the i-th line was
        clustered into.

        Args:
            centers (SetOfPoints) : a set of centers

        Returns:
            np.ndarray: an array of n indices, where each index is in the range [0,k-1]
        """

        assert self.get_size() > 0, "set is empty"
        centers_size = centers.get_size()
        assert centers_size > 0, "no centers given"

        self_size = self.get_size()
        dim = self.dim
        self_displacements = self.displacements
        self_spans = self.spans
        self_weights = self.weights
        centers_points = centers.points
        centers_weights = centers.weights

        centers_points_repeat_each_row = np.repeat(centers_points, self_size, axis=0).reshape(-1,
                                                                                              dim)  # this is a size*k-simensional vector, where the i-th element is center[j], where j=i/k
        a = np.where(np.isnan(centers_points))
        b = np.where(np.isnan(centers_points_repeat_each_row))
        a_flag = b_flag = False
        if np.sum(a) > 0:
            a_flag = True
        if np.sum(b) > 0:
            b_flag = True
        displacements_repeat_all = np.repeat(self_displacements.reshape(1, -1), centers_size, axis=0).reshape(-1,
                                                                                                              dim)  # repeating the displacement for the sum of squared distances calculation from each center for all the lines
        spans_repeat_all = np.repeat(self_spans.reshape(1, -1), centers_size, axis=0).reshape(-1,
                                                                                              dim)  # repeating the displacement for the sum of squared distances calculation from each center for all the lines
        centers_minus_displacements = centers_points_repeat_each_row - displacements_repeat_all
        centers_minus_displacements_squared_norms = np.sum(
            np.multiply(centers_minus_displacements, centers_minus_displacements), axis=1)
        centers_minus_displacements_dot_spans = np.multiply(centers_minus_displacements, spans_repeat_all)
        centers_minus_displacements_dot_spans_squared_norms = np.sum(
            np.multiply(centers_minus_displacements_dot_spans, centers_minus_displacements_dot_spans), axis=1)
        a = np.where(np.isnan(centers_minus_displacements_squared_norms))
        b = np.where(np.isnan(centers_minus_displacements_dot_spans_squared_norms))
        a_flag = b_flag = False
        if np.sum(a) > 0:
            a_flag = True
        if np.sum(b) > 0:
            b_flag = True
        all_unwighted_distances = centers_minus_displacements_squared_norms - centers_minus_displacements_dot_spans_squared_norms
        self_weights_repeat_all = np.repeat(self_weights.reshape(-1, 1), centers_size, axis=0).reshape(-1, 1)
        centers_weights_repeat_each_row = np.repeat(centers_weights, self_size, axis=0).reshape(-1, 1)
        total_weights = np.multiply(self_weights_repeat_all, centers_weights_repeat_each_row)
        all_weighted_distances = np.multiply(all_unwighted_distances.reshape(-1, 1), total_weights.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, self_size)

        # sum_of_squared_distances_reshaped = sum_of_squared_distances.reshape(-1,size)
        # sum_of_squared_distances_reshaped_mins = np.min(all_distances, axis=0) #this is a size-dimensional vector, where the i-th element contains the smallest distance from the i-th line to the given set of centers
        cluster_indices = np.argmin(all_distances.T,
                                    axis=1)  # the i-th element in this array contains the index of the cluster the i-th line was clusterd into.
        if np.min(all_distances) < 0:
            x = 2
        return cluster_indices

    ##################################################################################

    def get_centers_for_given_clusters(self, current_indices_cluster):
        """
        This method gets a size-dimensional vector, where size is the number of lines in the set, contains numbers in
        the range [0,k-1], that represent the cluster number that each line in the set was clustered into, and returns
        k center, one for each lines cluster, that minimizes the sum of squared distances in the set.

        Args:
            current_indices_cluster (np.ndarray) : list of indices in the range [0,k-1].

        Returns:
            np.ndarray: a set k centers that ninimizes the sum of squared distances to every line in each center's cluster
        """

        size = len(current_indices_cluster)
        assert size > 0, "set is empty"
        assert size == len(
            current_indices_cluster), "current_indices_cluster size is not the number of lines in the set"

        displacements = self.displacements
        dim = self.dim

        k = np.max(current_indices_cluster) + 1
        for i in range(k):
            indices_clustered_to_i = np.asarray(np.where(current_indices_cluster == i))[
                0]  # all the indices that contains i in current_indices_cluster
            displacements_clustered_to_i = displacements[
                indices_clustered_to_i]  # all the displacements in the i-th cluster
            cluster_i_center = np.mean(displacements_clustered_to_i, axis=0).reshape(-1,
                                                                                     dim)  # the center of cluster of lines, that minimizes the sum of squared distances to the lines in the cluster is the mean of the displacements in the cluster, under the assumption that each line is spanned by a unit vector and its displacements is the closest point in the line to the origin
            if i == 0:
                centers = cluster_i_center
            else:
                centers = np.concatenate((centers, cluster_i_center), axis=0).reshape(-1, dim)
        centers = centers.reshape(-1, dim)
        return centers

    ##################################################################################

    def get_sum_of_distances_to_centers(self, centers):
        """
        This method gets a cet of k points and return the sum of squared distances from these points to the lines in
        the set

        Args:
            centers (SetOfPoints) : a set of k centers

        Returns:
            float: the sum of squared distances to the lines in the set from the centers
        """

        assert self.get_size() > 0, "set is empty"
        centers_size = centers.get_size()
        assert centers_size > 0, "no centers given"

        dim = self.dim
        self_size = self.get_size()
        self_displacements = self.displacements
        self_spans = self.spans
        self_weights = self.weights

        centers_points = centers.points
        centers_weights = centers.weights

        centers_points_repeat_each_row = np.repeat(centers_points, self_size, axis=0).reshape(-1,
                                                                                              dim)  # this is a k*size array where every k points were duplicated size times
        self_displacements_repeat_all = np.repeat(self_displacements.reshape(1, -1), centers_size, axis=0).reshape(-1,
                                                                                                                   dim)  # this is a size*k array where every size displacements were duplicated k times
        self_spans_repeat_all = np.repeat(self_spans.reshape(1, -1), centers_size, axis=0).reshape(-1,
                                                                                                   dim)  # this is a size*k array where every size spans were duplicated k times
        self_weights_repeat_all = np.repeat(self_weights.reshape(1, -1), centers_size,
                                            axis=0)  # this is a size*k array where every size spans were duplicated k times
        centers_weights_repeat_each_row = np.repeat(centers_weights, self_size, axis=0).reshape(-1,
                                                                                                1)  # this is a size*k array where every size spans were duplicated k times
        centers_points_repeat_each_row_minus_displacements_repeat_all = centers_points_repeat_each_row - self_displacements_repeat_all
        centers_points_minus_displacements_norm_squared = np.sum(
            centers_points_repeat_each_row_minus_displacements_repeat_all ** 2, axis=1)
        centers_points_minus_displacements_mul_spans_norm_squared = np.sum(
            np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all, self_spans_repeat_all) ** 2,
            axis=1)
        unweighted_all_distances = centers_points_minus_displacements_norm_squared.reshape(-1,
                                                                                           1) - centers_points_minus_displacements_mul_spans_norm_squared.reshape(
            -1, 1)
        # for i in range(len(unweighted_all_distances)):
        #    if unweighted_all_distances[i] < 0:
        #        unweighted_all_distances[i] = 0
        total_weights = np.multiply(centers_weights_repeat_each_row.reshape(-1, 1),
                                    self_weights_repeat_all.reshape(-1, 1))
        all_weighted_distances = np.multiply(unweighted_all_distances.reshape(-1, 1), total_weights.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, self_size)
        """
        min_unweighted_all_distances = np.min(unweighted_all_distances)
        min_centers_weights_repeat_each_row = np.min(centers_weights_repeat_each_row)
        min_self_weights_repeat_all = np.min(self_weights_repeat_all)
        min_all_weighted_distances = np.min(all_weighted_distances)
        min_all_distances = np.min(all_distances)
        #plt.plot(unweighted_all_distances)
        j = 0
        for i in range(len(unweighted_all_distances)):
            if unweighted_all_distances[i] < 0:
                j+=1
        print(np.max(np.sort(unweighted_all_distances)))
        print(np.min(np.sort(unweighted_all_distances)))
        #plt.show()
        """
        all_distances_min = np.min(all_distances, axis=0)
        sum_of_squared_distances = np.sum(all_distances_min)
        if sum_of_squared_distances <= 0:
            x = 2
        return sum_of_squared_distances

    ##################################################################################

    def get_closest_lines_to_centers(self, centers, m, type):
        """
        Args:
            centers (npndarray) : d-dimensional points centers
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"
        Returns:
            SetOfLines: the lines that are closest to the given centers, by rate or by fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of lines in query is larger than number of lines in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) the rate invalid"

        self_spans = self.spans
        self_displacements = self.displacements
        self_weights = self.weights

        cluster_indices = self.get_indices_clusters(centers)
        centers_by_cluster_indices = centers.get_points_from_indices(
            cluster_indices)  # that is an array of size points, where the i-th element is the centers[cluster_indices[i]]

        centeres_clustered_points = centers_by_cluster_indices.points
        centeres_clustered_weights = centers_by_cluster_indices.weights

        centers_by_cluster_indices_minus_displacements = centeres_clustered_points - self_displacements
        centers_by_cluster_indices_minus_displacements_squared_norms = np.sum(
            np.multiply(centers_by_cluster_indices_minus_displacements, centers_by_cluster_indices_minus_displacements),
            axis=1)
        centers_mul_spans_squared_norms = np.sum(
            np.multiply(centers_by_cluster_indices_minus_displacements, self_spans) ** 2, axis=1)
        all_unweighted_distances = centers_by_cluster_indices_minus_displacements_squared_norms - centers_mul_spans_squared_norms
        total_weights = np.multiply(centeres_clustered_weights.reshape(-1, 1), self_weights.reshape(-1, 1)).reshape(-1)
        all_distances = np.multiply(all_unweighted_distances.reshape(-1, 1), total_weights.reshape(-1, 1)).reshape(-1)
        if type == "by rate":
            m = int(m * self.get_size())  # number of lines is m percents of size
            all_distances_mth_index_in_the_mth_place = np.argpartition(all_distances, m)
        first_m_smallest_distances_indices = all_distances_mth_index_in_the_mth_place[:m]
        spans_subset = self.spans[first_m_smallest_distances_indices]
        displacements_subset = self.displacements[first_m_smallest_distances_indices]
        weights_subset = self.weights[first_m_smallest_distances_indices]
        return SetOfLines(spans_subset, displacements_subset, weights_subset)

    ##################################################################################

    def get_farthest_lines_to_centers(self, centers, m, type):
        """
        Args:
            centers (npndarray) : d-dimensional points centers
            m (int): size of sample - may be percent or fixed number, depends on the parameter 'type'
            type (str): available values: "by number"/"by rate"
        Returns:
            SetOfLines: the lines that are farthest to the given centers, by rate or by fixed number
        """

        assert type == "by number" or type == "by rate", "type undefined"
        if type == "by number":
            assert m <= self.get_size(), "(1) Number of lines in query is larger than number of lines in the set"
        if type == "by rate":
            assert m >= 0 and m <= 1, "(2) the rate invalid"

        self_spans = self.spans
        self_displacements = self.displacements
        self_weights = self.weights

        cluster_indices = self.get_indices_clusters(centers)
        centers_by_cluster_indices = centers.get_points_from_indices(
            cluster_indices)  # that is an array of size points, where the i-th element is the centers[cluster_indices[i]]

        centeres_clustered_points = centers_by_cluster_indices.points
        centeres_clustered_weights = centers_by_cluster_indices.weights

        centers_by_cluster_indices_minus_displacements = centeres_clustered_points - self_displacements
        centers_by_cluster_indices_minus_displacements_squared_norms = np.sum(
            np.multiply(centers_by_cluster_indices_minus_displacements, centers_by_cluster_indices_minus_displacements),
            axis=1)
        centers_mul_spans_squared_norms = np.sum(
            np.multiply(centers_by_cluster_indices_minus_displacements, self_spans) ** 2, axis=1)
        all_unweighted_distances = centers_by_cluster_indices_minus_displacements_squared_norms - centers_mul_spans_squared_norms
        total_weights = np.multiply(centeres_clustered_weights.reshape(-1, 1), self_weights.reshape(-1, 1)).reshape(-1)
        all_distances = np.multiply(all_unweighted_distances.reshape(-1, 1), total_weights.reshape(-1, 1)).reshape(-1)
        if type == "by rate":
            m = int(m * self.get_size())  # number of lines is m percents of size
        # m_th_distance = np.partition(all_distances, m)[m]  # the m-th distance
        # distances_higher_than_median_indices = np.where(all_distances >= m_th_distance)  # all the m highest distances indices in all_distances
        all_distances_mth_index_in_the_mth_place = np.argpartition(all_distances, m)
        if len(all_distances_mth_index_in_the_mth_place) % 2 == 0:
            first_m_smallest_distances_indices = all_distances_mth_index_in_the_mth_place[m:len(all_distances)]
        else:
            first_m_smallest_distances_indices = all_distances_mth_index_in_the_mth_place[m - 1:len(all_distances)]
        spans_subset = self.spans[first_m_smallest_distances_indices]
        displacements_subset = self.displacements[first_m_smallest_distances_indices]
        weights_subset = self.weights[first_m_smallest_distances_indices]
        return SetOfLines(spans_subset, displacements_subset, weights_subset)

    ##################################################################################

    def get_lines_at_indices(self, indices):
        """
        Args:
            indices (list of ints) : list of indices.

        Returns:
            SetOfLines: a set of lines that contains the points in the input indices
        """

        assert self.get_size() > 0, "set is empty"
        assert len(indices) > 0, "no indices given"

        new_spans = self.spans[indices]
        new_displacements = self.displacements[indices]
        new_weights = self.weights[indices]

        L = SetOfLines(new_spans, new_displacements, new_weights)
        return L

    ##################################################################################

    def get_cost_to_projected_mean(self, centers):
        """
        This function gets a set of centers, project them on the lines, take the mean of the projected points, and returns
        the sum of squared distances from this mean to the projected points
        :param centers:
        :return:
        """

        spans = self.spans
        displacements = self.displacements
        dim = self.dim

        indices_cluster = self.get_indices_clusters(centers)
        centers_at_indices_cluster = centers[indices_cluster]
        centers_minus_displacements = centers_at_indices_cluster - displacements
        centers_minus_displacements_dot_spans = np.multiply(centers_minus_displacements, spans)
        projected_points = centers_minus_displacements_dot_spans + displacements
        missing_entries_indices = np.argmax(spans, axis=1)
        original_points = copy.deepcopy(displacements)
        original_points[:, missing_entries_indices] = projected_points[:, missing_entries_indices]
        the_mean = np.mean(projected_points, axis=0)
        cost = self.get_sum_of_distances_to_centers(the_mean.reshape(-1, dim))
        return cost

    ##################################################################################

    def add_set_of_lines(self, other):
        """
        TODO: complete
        :param other:
        :return:
        """

        if self.get_size() == 0:
            self.dim = copy.deepcopy(other.dim)
            self.spans = copy.deepcopy(other.spans)
            self.weights = copy.deepcopy(other.weights)
            self.displacements = copy.deepcopy(other.displacements)
            return
        self.spans = np.concatenate((self.spans, other.spans))
        # self.weights = np.concatenate((self.weights, other.weights))
        self.weights = np.concatenate((self.weights.reshape(-1, 1), other.weights.reshape(-1, 1)))
        self.displacements = np.concatenate((self.displacements, other.displacements))

    ##################################################################################

    def get_projected_centers(self, centers):
        """
        This function gets a set of k centers, project each one of the centers onto its closest line in the ser and
        returns the n projected centers
        :param centers:
        :return:
        """

        spans = self.spans
        displacements = self.displacements
        dim = self.dim

        indices_cluster = self.get_indices_clusters(centers)
        centers_at_indices_cluster = centers.get_points_from_indices(indices_cluster)
        centers_points_at_indices_cluster = centers_at_indices_cluster.points
        centers_minus_displacements = centers_points_at_indices_cluster - displacements
        centers_minus_displacements_dot_spans = np.multiply(centers_minus_displacements, spans)
        projected_points = centers_minus_displacements_dot_spans + displacements
        return projected_points

    ##################################################################################

    def get_lines_at_indexes_interval(self, start, end):
        """
        Args:
            start (int) : starting index
            end (end) : ending index

        Returns:
            SetOfLines: a set of lines that contains the points in the given range of indices
        """

        size = end - start
        indices = np.asarray(range(size)) + start

        spans_subset = self.spans[indices]
        displacements_subset = self.displacements[indices]
        weights_subset = self.weights[indices]
        return SetOfLines(spans_subset, displacements_subset, weights_subset)

    ##################################################################################

    def remove_lines_at_indexes(self, start, end):
        """
        TODO: complete
        :param start:
        :param end:
        :return:
        """
        indexes = np.arange(start, end)
        self.spans = np.delete(self.spans, indexes, axis=0)
        self.displacements = np.delete(self.displacements, indexes, axis=0)
        self.weights = np.delete(self.weights, indexes, axis=0)
        self.sensitivities = np.delete(self.sensitivities, indexes, axis=0)

    ##################################################################################

    def set_all_weights_to_specific_value(self, value):
        """
        TODO: complete
        :param value:
        :return:
        """

        new_weights = np.ones(self.get_size()) * value
        self.weights = new_weights

    ##################################################################################

    def normalized_lines_representation(self):
        """
        This method gets a set of n lines represented by an array of n spanning vectors and an array od n displacements
        vectors, and returns these spanning vectors normalized and change each displacements in each line to be the
        closest point on the line to the origin. It is required in order to calculate all the distances later.
        Args:
            spans (np.ndarray) : an array of spanning vectors
            displacements (np.ndarray) : an array of displacements vectors

        Returns:
            spans_normalized, displacements_closest_to_origin (np.ndarray, np,ndarray) : the spanning vectors and the
                displacements vectors normalized and moved as required.
        """
        spans = self.spans
        displacements = self.displacements

        assert len(spans) > 0, "assert no spanning vectors"
        assert len(displacements) > 0, "assert no displacements vectors"
        assert len(spans) == len(displacements), "number of spanning vectors and displacements vectors not equal"

        dim = np.shape(spans)[1]

        spans_norms = np.sqrt(np.sum(spans ** 2, axis=1))
        spans_norms_repeat = np.repeat(spans_norms, dim, axis=0).reshape(-1, dim)
        spans_normalized = spans / spans_norms_repeat
        for i in range(len(spans_normalized)):
            for j in range(len(spans_normalized[i])):
                if i == 87:
                    x = 2
                val = spans_normalized[i, j]
                if math.isnan(val):
                    spans_normalized[i, j] = 0

        # print("spans_normalized: \n", spans_normalized)
        # con = np.array([[]])
        displacements_minus_one = displacements * -1
        d = np.sum(np.multiply(displacements_minus_one, spans_normalized), axis=1)
        d_repeat = np.repeat(d, dim, axis=0).reshape(-1, dim)
        displacements_closest_to_origin = displacements + np.multiply(spans_normalized, d_repeat)
        """
        #displacements_mul_spans_normalized = np.multiply(displacements_minus_one, spans_normalized)
        for i in range(len(displacements_minus_one)):
            minus = (displacements_minus_one[i] - spans_normalized[i] * d [i]).reshape(-1, self.dim)
            plus =  (displacements_minus_one[i] + spans_normalized[i] * d [i]).reshape(-1, self.dim)
            minus_norm = np.sqrt(np.sum(minus ** 2, axis=1))
            plus_norm = np.sqrt(np.sum(plus ** 2, axis=1))
            if minus_norm < plus_norm:
                if i == 0:
                    con = minus.reshape(-1, self.dim)
                    continue
                con = np.append(con,minus,axis=0)
            else:
                if i == 0:
                    con = plus.reshape(-1, self.dim)
                    continue
                con = np.append(con,plus,axis=0)
        """
        self.displacements = displacements_closest_to_origin
        self.spans = spans_normalized
        # displacements_mul_minus_1 = displacements * -1
        # displacements_mul_minus_1_mul_spans_normalized = np.sum(np.multiply(displacements_mul_minus_1, spans_normalized), axis=1)
        # displacements_mul_minus_1_mul_spans_normalized_repeat_each_col = np.repeat(
        #    displacements_mul_minus_1_mul_spans_normalized, dim, axis=0).reshape(-1, dim)
        # disp_mul_spans_normalized = np.multiply(spans_normalized,
        #                                        displacements_mul_minus_1_mul_spans_normalized_repeat_each_col)
        # displacements_closest_to_origin = disp_mul_spans_normalized + displacements

        # print("displacements_closest_to_origin: \n", displacements_closest_to_origin)

    ##################################################################################

    def get_sensitivities_first_argument_for_centers(self, B):
        """

        :param B (SetOfPoints) :  a set of centers to compute the sensitivities first arfument as in Alg.4 in the paper
        :return (ndarray) : an array of n numbers, where the i-th number is the sensitivity first arg of the i-th line
        """

        assert B.get_size() > 0, "The number of centers is zero"

        cost_to_B = self.get_sum_of_distances_to_centers(B)

        cluster_indexes = self.get_indices_clusters(B)
        clustered_points = B.get_points_from_indices(cluster_indexes)

        dim = self.dim
        self_size = self.get_size()
        self_displacements = self.displacements
        self_spans = self.spans
        self_weights = self.weights

        centers_points = clustered_points.points
        centers_weights = clustered_points.weights

        centers_points_repeat_each_row = np.repeat(centers_points, self_size, axis=0).reshape(-1, dim)
        centers_weights_repeat_each_row = np.repeat(centers_weights, self_size, axis=0).reshape(-1, 1)
        self_displacements_repeat_all = np.repeat(self_displacements.reshape(1, -1), self_size, axis=0).reshape(-1, dim)
        self_spans_repeat_all = np.repeat(self_spans.reshape(1, -1), self_size, axis=0).reshape(-1, dim)
        self_weights_repeat_all = np.repeat(self_weights.reshape(1, -1), self_size, axis=0)
        centers_points_repeat_each_row_minus_displacements_repeat_all = centers_points_repeat_each_row - self_displacements_repeat_all
        # centers_points_minus_displacements_norm_squared = np.sum(np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all,centers_points_repeat_each_row_minus_displacements_repeat_all), axis=1)
        centers_points_minus_displacements_norm_squared = np.sum(
            centers_points_repeat_each_row_minus_displacements_repeat_all ** 2, axis=1)
        try:
            the_flag = False
            centers_points_minus_displacements_mul_spans_norm_squared = np.sum(
                np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all, self_spans_repeat_all) ** 2,
                axis=1)
        except:
            the_flag = True
            self_spans_repeat_all_nan_indexes = np.where(np.isnan(self_spans_repeat_all))
            self_spans_repeat_all[self_spans_repeat_all_nan_indexes] = np.inf
            centers_points_minus_displacements_mul_spans_norm_squared = np.sum(
                np.multiply(centers_points_repeat_each_row_minus_displacements_repeat_all, self_spans_repeat_all) ** 2,
                axis=1)
            x = 2
        unweighted_all_distances = centers_points_minus_displacements_norm_squared - centers_points_minus_displacements_mul_spans_norm_squared
        less_than_zero_indexes = np.where(unweighted_all_distances < 0)
        is_nan_indexes = np.where(np.isnan(unweighted_all_distances))
        is_inf_indexes = np.where(np.isinf(unweighted_all_distances))
        less_than_zero_sum = np.sum(less_than_zero_indexes)
        is_nan_indexes_sum = np.sum(is_nan_indexes)
        is_inf_indexes_sum = np.sum(is_inf_indexes)
        if less_than_zero_sum + is_nan_indexes_sum + is_inf_indexes_sum > 0:
            print("less_than_zero_sum: ", less_than_zero_sum)
            x = 2
        # for i in range(len(unweighted_all_distances)):
        #    if unweighted_all_distances[i] < 0:
        #        unweighted_all_distances[i] = 0
        total_weights = np.multiply(centers_weights_repeat_each_row.reshape(-1, 1),
                                    self_weights_repeat_all.reshape(-1, 1))
        all_weighted_distances = np.multiply(unweighted_all_distances.reshape(-1, 1), total_weights.reshape(-1, 1))
        all_distances = (all_weighted_distances).reshape(-1, self_size)
        all_distances_min = np.min(all_distances, axis=0)
        sensitivities_first_argument = all_distances_min / cost_to_B

        return sensitivities_first_argument

    def shuffle_lines(self):
        """
        This method shuffles the lines in the set
        :return:
        """

        random_indexes = np.random.permutation(self.get_size())
        self.spans = self.spans[random_indexes]
        self.displacements = self.displacements[random_indexes]
        self.weights = self.weights[random_indexes]

    def normalize_spans(self):
        spans_norm = np.sum(self.spans ** 2, axis=1) ** 0.5
        spans_norm_inv = 1 / spans_norm
        spans_norm_inv_repeated = np.repeat(spans_norm_inv.reshape(-1), self.dim).reshape(-1, self.dim)
        self.spans = np.multiply(self.spans, spans_norm_inv_repeated)

    def multiply_weights_by_value(self, val):
        self.weights = self.weights * val



