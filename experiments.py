import io
import time

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn import svm
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import Normalizer

#to ensure the same splits in cross validation
random_seed = 7

#positive text
pos = []
#negative text
neg = []
#test set text
test = []

#open files
positive = io.open('data_stopwords.pos', encoding = "ISO-8859-1")
negative = io.open('data_stopwords.neg', encoding = "ISO-8859-1")
testSet = io.open('data_stopwords.test', encoding = "ISO-8859-1")

#make sure we remove any new line characters
for line in positive:
    line = line.strip()
    pos.append(line)

for line in negative:
    line = line.strip()
    neg.append(line)

for line in testSet:
    line = line.strip()
    test.append(line)


#If we only want a small partition of the data use these two lines
#good to use if code is taking too long on full dataset
#pos = pos[0:3000]
#neg = neg[0:3000]

#put positive and negative sentiment together for train data
trainData = pos + neg
#build list of labels
trainLabels = [1]*len(pos) + [0]*len(neg)


#Vectorizing data based on booleans
#Tutorial code helped: https://colab.research.google.com/drive/1LQuuM9oNuQhX16jyMoD2ekkIvJ4nefHd#scrollTo=LZ4ftpnjRbUP


tfidf_pipeline = True
# Put the feature matrix in the *_matrix variables
# tfidf pipeline
if tfidf_pipeline:
    count_vect = CountVectorizer(binary = False,ngram_range=(1, 1)).fit(trainData);
    X_train_counts = count_vect.transform(trainData)
    X_test_counts = count_vect.transform(test)


    tfidf_transformer = TfidfTransformer().fit(X_train_counts)
    X_train_matrix = tfidf_transformer.transform(X_train_counts)
    X_test_matrix = tfidf_transformer.transform(X_test_counts)
# binary occurrences pipeline
if not tfidf_pipeline:
    count_vect = CountVectorizer(binary = True,ngram_range=(1, 1)).fit(trainData);
    X_train_matrix = count_vect.transform(trainData)
    X_test_matrix = count_vect.transform(test)

#setup the crossvalidation
kf = KFold(n_splits=5, shuffle = True, random_state = random_seed);

#choose model
#NOTE: SVM is slow

#model = MultinomialNB()

print("Try LogisticRegression with tfidf")
log_model = linear_model.LogisticRegression()
#perform cross validation, see accuracy on each fold and average accuracy
log_accuracy = model_selection.cross_val_score(log_model, X_train_matrix, trainLabels, cv=kf)
print(log_accuracy)
print(log_accuracy.mean())


print("Try SVM with tfidf")
# Do cross validation on the SVM with tfidf
svm_model = svm.SVC(kernel='linear')
svm_accuracy = model_selection.cross_val_score(svm_model, X_train_matrix, trainLabels, cv=kf)
print(svm_accuracy)
print(svm_accuracy.mean())


print("Try DecisionTree with tfidf")
tree_model = DecisionTreeClassifier()
tree_accuracy = model_selection.cross_val_score(svm_model, X_train_matrix, trainLabels, cv=kf)
print(tree_accuracy)
print(tree_accuracy.mean())
#Set to true if you want to produce a submission csv
csv = True

# set the model to be used for submission
model = svm_model
#CODE FOR BUILDING SUBMISSION

if csv:
    #open submission file
    submissionFile = open('submission.csv','w')

    #build model on full training data
    pred = model.fit(X_train_matrix, trainLabels)
    #use model to predict test data
    y_pred = pred.predict(X_test_matrix)

    #predictions array
    print(y_pred)

    #printing the sum of the predicition array
    #this should be ~12,500 as the test set is balanced
    #this a good thing to check to make sure your model is not biased
    print(sum(y_pred))

    #header for csv
    submissionFile.write('Id,Category')


    #There is definitly a better way to do this
    #I'm not sure whether Kaggle needs the answers to be in numerical order? 0,1,2,3,4.....

    #rebuild all the text files names from test and sort them
    #we do this because in dataProcess the files weren't read in 0,1,2,3...
    #the file were read in 0,1,10,100,1000....
    files = []
    for i in range(len(y_pred)):
        files.append(str(i)+'.txt')
    files = sorted(files)

    #now break down the files names to just the number and pair with the prediction
    #our list is now [(0,pred),(1,pred),(10,pred),(100,pred)....]
    output = []
    for i in range(len(y_pred)):
        output.append([int(files[i].split('.')[0]),y_pred[i]])
    #sort this list
    #our list is now [(0,pred),(1,pred),(2,pred),(3,pred)....]
    output = sorted(output)

    #write these index prediction pairs to the csv file
    for i in range(len(y_pred)):
        submissionFile.write('\n')
        submissionFile.write(str(output[i][0])+','+str(output[i][1]))

    submissionFile.close()
