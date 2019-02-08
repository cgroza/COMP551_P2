import numpy
# Here we will put the text preprocessing code that will load the data and generate the features.

class PreprocessTrainingData:
    def __init__(self, pos_dir, neg_dir):
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir

    def numpy_feature_matrix(self):
        """
        Returns a numpy matrix with the features as columns and examples as rows.
        """
        pass

    def numpy_target_vector(slef):
        """
        Returns a numpy array with the true classes of the training points.
        """
        pass

class PreprocessTestingData:
    def __init__(self, testing_dir):
        self.testing_dir = testing_dir
