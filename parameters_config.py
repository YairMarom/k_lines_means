class ParameterConfig:
    def __init__(self):
        # main parameters
        self.header_indexes = None
        self.dim = None
        self.lines_number = None

        # experiment  parameters
        self.sample_sizes = None
        self.inner_iterations = None
        self.centers_number = None
        self.outliers_trashold_value = None

        # EM k means for lines estimator parameters
        self.multiplications_of_k = None
        self.EM_iteration_test_multiplications = None

        # EM k means for points estimator parameters
        self.ground_true_iterations_number_ractor = None

        # k means for lines coreset parameters
        self.inner_a_b_approx_iterations = None
        self.sample_rate_for_a_b_approx = None

        # weighted centers coreset parameters
        self.median_sample_size = None
        self.closest_to_median_rate = None
        self.number_of_remains_multiply_factor = None
        self.max_sensitivity_multiply_factor = None

        # iterations
        self.RANSAC_iterations = None
        self.coreset_iterations = None
        self.RANSAC_EM_ITERATIONS = None
        self.coreset_to_ransac_time_rate = None

        # files
        self.input_points_file_name = None

        #missing entries parameters
        self.missing_entries_alg = None
        self.cost_type= None
        self.KNN_k = None


        #data handler parameters
        self.points_number = None
        self.output_file_name = None

