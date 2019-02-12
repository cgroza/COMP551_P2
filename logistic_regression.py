import data_loader
import numpy
import word_features
import sklearn.linear_model as models
import sklearn.metrics as metrics

(total, pos, neg) = word_features.get_training_words()

training_data = data_loader.LoadTrainingData("train/pos", "train/neg")
# create feature extractor on the data and the word list
feature_extractor = data_loader.ExtractFeatures(training_data.data, total)
# partition into 10 sets
binary_feature = feature_extractor.extract_binary()
data_partitions = feature_extractor.partition(binary_feature, feature_extractor.class_vector)

# save memory
del binary_feature
del training_data


# Do 10-cross validation
accuracies = []
# first set for testing
for i in range(0, 10):
    test = data_partitions[i]
    # the other 9 for training
    training_pairs = data_partitions[0:i] + data_partitions[i + 1:]
    assert len(training_pairs) == 9
    # concatenate training matrices and vectors
    matrices = []
    vectors = []
    for matrix_vector_pair in training_pairs:
        matrices.append(matrix_vector_pair[0])
        vectors.append(matrix_vector_pair[1])

    training = numpy.concatenate(matrices)
    # save memory
    del matrices
    class_labels = numpy.concatenate(vectors)

    logistic_model = models.LogisticRegression(max_iter = 10000 ).fit(training, class_labels)

    predicted = logistic_model.predict(test[0])
    actual = test[1]
    print("Test i=" + str(i))
    print(metrics.accuracy_score(actual, predicted))
    print(predicted)
    print(actual)
    accuracies.append(metrics.accuracy_score(actual, predicted))

print("Average accuracy of logistic regression with binary features:")
print(sum(accuracies)/len(accuracies))
import sklearn.metrics as metrics
