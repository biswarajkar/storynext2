# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
#
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn


import sys
import os
import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


#
# def usage():
#     print("Usage:")
#     print("python %s ./Dataset" % sys.argv[0])


def run(dir):
    # if len(sys.argv) < 2:
    #     usage()
    #     sys.exit(1)

    data_dir = dir
    classes = ['pos', 'neg']
    folders = ['test', 'train']

    # Read the data
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    for cur_folder in folders:
        for curr_class in classes:
            dirname = os.path.join(data_dir, cur_folder, curr_class)
            print("dirname:", dirname)
            for fname in os.listdir(dirname):
                print(fname)
                with open(os.path.join(dirname, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if cur_folder == 'test':
                        test_data.append(content)
                        test_labels.append(curr_class)
                    else:
                        train_data.append(content)
                        train_labels.append(curr_class)

    print(len(test_data))
    print(len(train_data))
    # Create feature vectors
    # min_df = 5, discard words appearing in less than 5 documents
    # max_df = 0.8, discard words appering in more than 80 % of the documents
    # sublinear_tf = True, use sublinear weighting
    # use_idf = True, enable IDF
    vectorizer = TfidfVectorizer(max_features=40000,
                                 min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # # Perform classification with SVM, kernel=rbf
    # classifier_rbf = svm.SVC()
    # t0 = time.time()
    # classifier_rbf.fit(train_vectors, train_labels)
    # t1 = time.time()
    # prediction_rbf = classifier_rbf.predict(test_vectors)
    # t2 = time.time()
    # time_rbf_train = t1 - t0
    # time_rbf_predict = t2 - t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1 - t0
    time_linear_predict = t2 - t1

    # # Perform classification with SVM, kernel=linear
    # classifier_liblinear = svm.LinearSVC()
    # t0 = time.time()
    # classifier_liblinear.fit(train_vectors, train_labels)
    # t1 = time.time()
    # prediction_liblinear = classifier_liblinear.predict(test_vectors)
    # t2 = time.time()
    # time_liblinear_train = t1 - t0
    # time_liblinear_predict = t2 - t1

    # Perform classification with RandomForest
    classifier_rdf = RandomForestClassifier(n_estimators = 100, n_jobs = -1, verbose = 1)
    t0 = time.time()
    classifier_rdf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rdf = classifier_rdf.predict(test_vectors)
    t2 = time.time()
    time_rdf_train = t1 - t0
    time_rdf_predict = t2 - t1

    # Print results in a nice table
    # print("Results for SVC(kernel=rbf)")
    # print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    # print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    # print("Results for LinearSVC()")
    # print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    # print(classification_report(test_labels, prediction_liblinear))
    print("Results for RandomForest")
    print("Training time: %fs; Prediction time: %fs" % (time_rdf_train, time_rdf_predict))
    print(classification_report(test_labels, prediction_rdf))

# Total:
# Results for SVC(kernel=linear)
# Training time: 5.282428s; Prediction time: 0.269310s
#              precision    recall  f1-score   support
#
#         neg       0.60      0.53      0.56        75
#         pos       0.58      0.64      0.61        76
#
# avg / total       0.59      0.59      0.59       151
#
# Results for RandomForest
# Training time: 0.849757s; Prediction time: 0.106429s
#              precision    recall  f1-score   support
#
#         neg       0.62      0.32      0.42        75
#         pos       0.54      0.80      0.65        76
#
# avg / total       0.58      0.56      0.54       151

# Sport:
# Results for SVC(kernel=linear)
# Training time: 4.865517s; Prediction time: 0.220716s
#              precision    recall  f1-score   support
#
#         neg       0.78      0.53      0.63        75
#         pos       0.29      0.56      0.38        25
#
# avg / total       0.66      0.54      0.57       100
#
# Results for RandomForest
# Training time: 0.736891s; Prediction time: 0.106315s
#              precision    recall  f1-score   support
#
#         neg       0.75      0.36      0.49        75
#         pos       0.25      0.64      0.36        25
#
# avg / total       0.62      0.43      0.45       100

# Literrature:
# Results for SVC(kernel=linear)
# Training time: 4.982864s; Prediction time: 0.171225s
#              precision    recall  f1-score   support
#
#         neg       0.77      0.53      0.63        75
#         pos       0.29      0.54      0.37        26
#
# avg / total       0.64      0.53      0.56       101
#
# Results for RandomForest
# Training time: 0.801408s; Prediction time: 0.105362s
#              precision    recall  f1-score   support
#
#         neg       0.78      0.24      0.37        75
#         pos       0.27      0.81      0.40        26
#
# avg / total       0.65      0.39      0.38       101

# Politics:
# Results for SVC(kernel=linear)
# Training time: 5.652754s; Prediction time: 0.211159s
#              precision    recall  f1-score   support
#
#         neg       0.91      0.53      0.67        75
#         pos       0.38      0.84      0.52        25
#
# avg / total       0.78      0.61      0.63       100
#
# Results for RandomForest
# Training time: 0.782652s; Prediction time: 0.113840s
#              precision    recall  f1-score   support
#
#         neg       0.90      0.25      0.40        75
#         pos       0.29      0.92      0.44        25
#
# avg / total       0.75      0.42      0.41       100