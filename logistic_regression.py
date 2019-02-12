import data_loader
import numpy
import word_features
import sklearn.linear_model as models

(total, pos, neg) = word_features.get_training_words()

training_data = data_loader.LoadTrainingData("train/pos", "train/neg")
# create feature extractor on the data and the word list
feature_extractor = data_loader.ExtractFeatures(training_data.data, total)
# partition into 10 sets
binary_feature = feature_extractor.extract_binary()
data_partitions = feature_extractor.partition(binary_feature, feature_extractor.class_vector)

#NOTE: we will do proper cross validation on this later
# first set for testing
test = data_partitions[0]
# the other 9 for training
training_pairs = data_partitions[1:]
# concatenate training matrices and vectors
matrices = []
vectors = []
for matrix_vector_pair in training_pairs:
    matrices.append(matrix_vector_pair[0])
    vectors.append(matrix_vector_pair[1])

training = numpy.concatenate(matrices)
class_labels = numpy.concatenate(vectors)


logistic_model = models.LogisticRegression().fit(training, class_labels)
