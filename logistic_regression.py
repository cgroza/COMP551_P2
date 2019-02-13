import data_loader
import os
import numpy
import sklearn.linear_model as models
import utility

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

def predict_testing(training_matrix, training_target):
    """
    Trains a model with the full training set and predicts the testing set
    """
    final_model = models.LogisticRegression(max_iter = 10000 ).fit(training_matrix, training_target)

    testing_data = LoadTestingData("test")
    if os.path.exists("testing_matrix.npy"):
        testing_matrix = numpy.load("testing_matrix.npy")
    else:
        feature_extractor = data_loader.ExtractFeatures(testing_data.data, shuffle = False)
        testing_matrix = feature_extractor.extract_tfidf()
        numpy.save("testing_matrix.npy", testing_matrix)

    predicted_testing = final_model.predict(testing_matrix)
    return([review['ex'] for review in testing_data.data], predicted_testing)
