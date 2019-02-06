#################################################################
#     Corset for Weighted centers of points                     #
#     Paper: http://people.csail.mit.edu/dannyf/outliers.pdf    #
#     Implemented by Yair Marom. yairmrm@gmail.com              #
#################################################################




from __future__ import division

import copy
import csv
import time

import numpy as np

from coreset_for_k_means_for_lines import CorsetForKMeansForLines
from coreset_node import CoresetNode

#from parameters_config import ParameterConfig
#from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute


"""
Class that performs the full streaming operation. Each step read m points from a file, comprase it and add it to the
coreset tree.
Attributes:
    stack (list): A list that simulates the streaming comprassion tree operations
    st (string): The path to the file
    m (int): size of chunk
    co (int): flag/number of points for read
    eps (float): error parameter
    delta (float): failure probability
"""


class CoresetStreamer:

    def __init__(self, sample_size, lines_number, k, parameters_config):

        self.stack = []
        self.k = k
        self.file_name = parameters_config.input_points_file_name
        self.sample_size = sample_size
        self.lines_number = lines_number #if points_number == -1 then it will read until EOF
        self.parameters_config = parameters_config

    ######################################################################

    def stream(self, L, is_spark_test = False):
        """
        The method start to get in a streaming points from the file st as required
        TODO: complete parameteres
        """
        coreset_starting_time = time.time()
        batch_size = self.sample_size*2
        starting_index = 0
        number_of_lines_read_so_far = 0
        Q = copy.deepcopy(L)
        while True:
            if number_of_lines_read_so_far == self.lines_number:
                break
            #if number_of_lines_read_so_far % int(self.lines_number / 10) == 0:
                #print("Lines read so far: ", number_of_lines_read_so_far)
                #sum_of_weights = 0
                #for t in range(len(self.stack)):
                #    sum_of_weights += np.sum(self.stack[t].points.weights)
                #print("Sum of weights so far: ", sum_of_weights)
                #print(" ")
            Q_size = Q.get_size()
            if batch_size > Q_size:
                self.add_to_tree(Q)
                break
            current_batch = Q.get_lines_at_indexes_interval(starting_index, starting_index + batch_size)
            Q.remove_lines_at_indexes(starting_index, starting_index+batch_size)
            self.add_to_tree(current_batch)
            number_of_lines_read_so_far += batch_size
        while len(self.stack) > 1:
            node1 = self.stack.pop()
            node2 = self.stack.pop()
            new_node = self.merge_two_nodes(node1, node2)
            self.stack.append(new_node)
        C = self.stack[0].lines
        coreset_ending_time = time.time()
        print("coreset sum of weights: ", np.sum(C.weights))
        if is_spark_test:
            return coreset_starting_time, coreset_ending_time
        return C, coreset_starting_time, coreset_ending_time

    ######################################################################

    def add_to_tree(self, L):
        L_size = L.get_size()
        if L_size > self.sample_size:
            coreset = CorsetForKMeansForLines(self.parameters_config).coreset(L=L, k=self.k, m=self.sample_size)
            x = np.sum(coreset.weights)
            current_node = CoresetNode(coreset)
        else:
            current_node = CoresetNode(L)

        if len(self.stack) == 0:
            self.stack.append(current_node)
            return

        stack_top_node = self.stack[-1]
        if stack_top_node.rank != current_node.rank:
            self.stack.append(current_node)
            return
        else:
            while stack_top_node.rank == current_node.rank: #TODO: take care for the case they are not equal, currently the node deosn't appanded to the tree
                self.stack.pop()
                current_node_sum_of_weights = np.sum(current_node.lines.weights)
                top_node_sum_of_weights = np.sum(stack_top_node.lines.weights)
                current_node = self.merge_two_nodes(current_node, stack_top_node)
                current_node_sum_of_weights = np.sum(current_node.lines.weights)
                if len(self.stack) == 0:
                    self.stack.append(current_node)
                    return
                stack_top_node = self.stack[-1]
                if stack_top_node.rank != current_node.rank:
                    self.stack.append(current_node)
                    return

    ######################################################################

    def merge_two_nodes(self, node1, node2):
        """
        The method gets two nodes of the corset tree, merge them, and return the corset of the merged nodes
        :param current_node: CoresetNode
        :param stack_top_node: CoresetNode
        """
        L1 = node1.lines
        L1_sum_of_weights = np.sum(L1.weights)
        L2 = node2.lines
        L2_sum_of_weights = np.sum(L2.weights)
        L1.add_set_of_lines(L2)
        coreset = CorsetForKMeansForLines(self.parameters_config).coreset(L=L1, k=self.k, m=self.sample_size)
        coreset_sum_of_weights = np.sum(coreset.weights)
        return CoresetNode(coreset, node1.rank+1)

    ######################################################################

    def create_synthetic_points(file_name, points):
        with open(file_name, "w") as csv_file:
            writer = csv.writer(csv_file, delimiter=',')
            for point in points:
                writer.writerow(point)
