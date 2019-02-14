import word_features
import data_loader
import numpy
import os
import sklearn.metrics as metrics

def compute_binary_training_matrix():
    if not os.path.exists("binary_training_matrix.npy"):
        (total, pos, neg) = word_features.get_training_words()

        training_data = data_loader.LoadTrainingData("train/pos", "train/neg")
        # create feature extractor on the data and the word list
        feature_extractor = data_loader.ExtractFeatures(training_data.data, total)
        # partition into 10 sets
        binary_feature = feature_extractor.extract_binary()
        numpy.save("binary_training_matrix.npy", binary_feature)
        class_vector = feature_extractor.class_vector
        numpy.save("binary_class_vector.npy", class_vector)
    else:
        binary_feature = numpy.load("binary_training_matrix.npy")
        class_vector = numpy.load("binary_class_vector.npy")
    return (binary_feature, class_vector)

def compute_tfidf_training_matrix():
    if not os.path.exists("tfidf_training_matrix.npy"):
        (total, pos, neg) = word_features.get_training_words()

        training_data = data_loader.LoadTrainingData("train/pos", "train/neg")
        # create feature extractor on the data and the word list
        feature_extractor = data_loader.ExtractFeatures(training_data.data, total)
        # partition into 10 sets
        tfidf_feature = feature_extractor.extract_tfidf()
        numpy.save("tfidf_training_matrix.npy", tfidf_feature)
        class_vector = feature_extractor.class_vector
        numpy.save("tfidf_class_vector.npy", class_vector)
    else:
        tfidf_feature = numpy.load("tfidf_training_matrix.npy")
        class_vector = numpy.load("tfidf_class_vector.npy")
    return (tfidf_feature, class_vector)

def k_fold_validate(feature_matrix, target_vector, Model):
    data_partitions = data_loader.ExtractFeatures.partition(feature_matrix, target_vector)
    # Do 10-cross validation
    accuracies = []
    # first set for testing
    for i in range(0, 10):
        test = data_partitions[i]
        # the other 9 for training
        training_pairs = data_partitions[0:i] + data_partitions[i + 1:]
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

        logistic_model = Model(max_iter = 10000 ).fit(training, class_labels)

        predicted = logistic_model.predict(test[0])
        actual = test[1]
        print("Test i=" + str(i))
        print(metrics.accuracy_score(actual, predicted))
        print(predicted)
        print(actual)
        accuracies.append(metrics.accuracy_score(actual, predicted))

    print("Average accuracy:")
    print(sum(accuracies)/len(accuracies))

def predict_testing(training_matrix, training_target, out, Model):
    """
    Trains a model with the full training set and predicts the testing set
    """
    final_model = Model(max_iter = 10000).fit(training_matrix, training_target)

    testing_data = data_loader.LoadTestingData("test")
    if os.path.exists("testing_matrix.npy"):
        testing_matrix = numpy.load("testing_matrix.npy")
    else:
        (total, pos, neg) = word_features.get_training_words()
        feature_extractor = data_loader.ExtractFeatures(testing_data.data, total, shuffle = False)
        testing_matrix = feature_extractor.extract_tfidf()
        numpy.save("testing_matrix.npy", testing_matrix)

    predicted_testing = final_model.predict(testing_matrix)
    with open(out, "w") as f:
        f.write("Id,Category\n")
        for i in range(len(predicted_testing)):
            f.write(testing_data.data[i]["ex"].replace(".txt", "") + "," + str(predicted_testing[i]) + "\n")
