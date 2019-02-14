import data_loader
import os
import numpy
import sklearn.linear_model as models
import utility
import word_features

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


utility.predict_testing(tfidf_train_data, tfidf_target, "logistic_pred.csv", models.LogisticRegression)
