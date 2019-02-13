import numpy
import sklearn.linear_model as models
import sklearn.metrics as metrics
import utility

# save memory
# del binary_feature
# del training_data


def train_final_model():
    final_model = models.LogisticRegression(max_iter = 10000 ).fit(binary_feature, feature_extractor.class_vector)

(train_data, target) = utility.compute_binary_training_matrix()

# k-fold-validate LogisticRegression with binary features
k_fold_validate(train_data, target, models.LogisticRegression)
