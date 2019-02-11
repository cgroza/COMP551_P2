import numpy
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist
# Here we will put the text preprocessing code that will load the data and generate the features.

# Citation for Lexicon 1
# This file and the papers can all be downloaded from
#    http://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html
#
# If you use this list, please cite the following paper:
#
#   Minqing Hu and Bing Liu. "Mining and Summarizing Customer Reviews."
#       Proceedings of the ACM SIGKDD International Conference on Knowledge
#       Discovery and Data Mining (KDD-2004), Aug 22-25, 2004, Seattle,
#       Washington, USA,

class ProcessText:
    def __init__(self):
        self._lematizer = nltk.WordNetLemmatizer()

    def process_text(self, txt_name, text, clss = None):
        # Here, we remove stop words
        word_list = [self._lematizer.lemmatize(word) for word in nltk.word_tokenize(text)
                     if word.lower() not in stopwords.words('english')]
        return {"ex": txt_name, "review": word_list, "class" : clss}

class ExtractFeatures:
    def __init__(self, data, word_list):
        """
        data: list of reviews to extract features from
        word_list: words to be used as features in the classification
        """

        # Words for which to collect features
        self.word_list = word_list
        self.data = data
        self.feature_matrix = None
        self.class_vector = None

    def extract_counts(self):
        matrix = []
        vector = []
        for review in self.data:
            features = []
            # Bias term
            features.append(1)
            # Add word counts in the order they are in the list
            for word in self.word_list:
                features.append(review["review"].count(word))
            matrix.append(features)
            vector.append(review["class"])
        self.feature_matrix = numpy.array(matrix)
        self.class_vector = numpy.array(vector)

    def extract_binary(self):
        matrix = []
        for example in self.feature_matrix[0:-1]:
            binary_features = []
            for entry in example:
                if entry > 0: binary_features.append(1)
                else: binary_features.append(0)
            matrix.append(binary_features)
        return numpy.array(matrix)

    def exract_tfidf(self):
        pass

class LoadTrainingData:
    def __init__(self, pos_dir, neg_dir):
        self.pos_dir = pos_dir
        self.neg_dir = neg_dir

        text_processor = ProcessText()
        self.data = []

        # Read each example in the negative training directory.
        # for txt_name in os.listdir(neg_dir):
        #     with open(os.path.join(neg_dir, txt_name)) as f:
        #         text = f.read()
        #         self.data.append(text_processor.process_text(txt_name, text, 0))

        # # Read each example in the positive training directory.
        # for txt_name in os.listdir(pos_dir):
        #     with open(os.path.join(pos_dir, txt_name)) as f:
        #         text = f.read()
        #         self.data.append(text_processor.process_text(txt_name, text, 1))

        # Or load preprocessed training data

        with open("training_data.data") as f:
            self.data = eval(f.read())

        # This is long to compute. We must save this to a file.
        # all_words = []
        # for review in self.data:
        #     all_words = all_words + review["review"]

        with open("word_freqs.data") as freqs:
            self.words_freq = eval(freqs.read())

class LoadTestingData:
    def __init__(self, testing_dir):
        self.testing_dir = testing_dir
        self.data = []

        for txt_name in os.listdir(testing_dir):
            with open(os.path.join(testing_dir, txt_name)) as f:
                text = f.read()
                self.data.append(text_processor.process_text(txt_name, text, None))


# Example of use
if __name__ == "__main__":
    training_data = LoadTrainingData("train/pos", "train/neg")
    # list of dictionaries {"ex" : file_name, "review" : word_list, "class" : example_class}
    training_data.data
    # frequency of words
    training_data.words_freq

    testing_data = LoadTestingData("test")
    # list of dictionaries {"ex" : file_name, "review" : word_list, "class" : None}
    testing_data.data
