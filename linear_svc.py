import sklearn.svm as svm
import utility

print("Tfidf features 10-fold validation")
print("Loading tfidf matrix")
(tfidf_train_data, tfidf_target) = utility.compute_tfidf_training_matrix()
# k-fold-validate LogisticRegression with tfidf features
utility.k_fold_validate(tfidf_train_data, tfidf_target, svm.LinearSVC)

