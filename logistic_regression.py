import numpy
import sklearn.linear_model as models
import utility

# save memory
# del binary_feature
# del training_data


def train_final_model():
    final_model = models.LogisticRegression(max_iter = 10000 ).fit(binary_feature, feature_extractor.class_vector)


print("Binary features 10-fold validation")
print("Loading binary matrix")
(binary_train_data, binary_target) = utility.compute_binary_training_matrix()
# k-fold-validate LogisticRegression with binary features
utility.k_fold_validate(binary_train_data, binary_target, models.LogisticRegression)

print("Tfidf features 10-fold validation")
print("Loading tfidf matrix")
(tfidf_train_data, tfidf_target) = utility.compute_tfidf_training_matrix()
# k-fold-validate LogisticRegression with tfidf features
utility.k_fold_validate(tfidf_train_data, tfidf_target, models.LogisticRegression)
