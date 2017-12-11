# You need to install scikit-learn:
# sudo pip install scikit-learn
#
# Dataset: Polarity dataset v2.0
# http://www.cs.cornell.edu/people/pabo/movie-review-data/
# Full discussion:
# https://marcobonzanini.wordpress.com/2015/01/19/sentiment-analysis-with-python-and-scikit-learn


import time
import os
from sklearn import svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report


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
            for fname in os.listdir(dirname):
                with open(os.path.join(dirname, fname), 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    if cur_folder == 'test':
                        test_data.append(content)
                        test_labels.append(curr_class)
                    else:
                        train_data.append(content)
                        train_labels.append(curr_class)

    # Create feature vectors
    # min_df = 5, discard words appearing in less than 5 documents
    # max_df = 0.8, discard words appering in more than 80 % of the documents
    # sublinear_tf = True, use sublinear weighting
    # use_idf = True, enable IDF
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(train_data)
    test_vectors = vectorizer.transform(test_data)

    # Perform classification with SVM, kernel=rbf
    classifier_rbf = svm.SVC()
    t0 = time.time()
    classifier_rbf.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1 - t0
    time_rbf_predict = t2 - t1

    # Perform classification with SVM, kernel=linear
    classifier_linear = svm.SVC(kernel='linear')
    t0 = time.time()
    classifier_linear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_linear = classifier_linear.predict(test_vectors)
    t2 = time.time()
    time_linear_train = t1 - t0
    time_linear_predict = t2 - t1

    # Perform classification with SVM, kernel=linear
    classifier_liblinear = svm.LinearSVC()
    t0 = time.time()
    classifier_liblinear.fit(train_vectors, train_labels)
    t1 = time.time()
    prediction_liblinear = classifier_liblinear.predict(test_vectors)
    t2 = time.time()
    time_liblinear_train = t1 - t0
    time_liblinear_predict = t2 - t1

    # Print results in a nice table
    print("Results for SVC(kernel=rbf)")
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(test_labels, prediction_rbf))
    print("Results for SVC(kernel=linear)")
    print("Training time: %fs; Prediction time: %fs" % (time_linear_train, time_linear_predict))
    print(classification_report(test_labels, prediction_linear))
    print("Results for LinearSVC()")
    print("Training time: %fs; Prediction time: %fs" % (time_liblinear_train, time_liblinear_predict))
    print(classification_report(test_labels, prediction_liblinear))

    # Results for SVC(kernel=rbf)
    # Training time: 5.164954s; Prediction time: 0.360977s
    #              precision    recall  f1-score   support
    #
    #         neg       0.53      0.25      0.34        75
    #         pos       0.51      0.78      0.62        76
    #
    # avg / total       0.52      0.52      0.48       151
    #
    # Results for SVC(kernel=linear)
    # Training time: 5.006144s; Prediction time: 0.296542s
    #              precision    recall  f1-score   support
    #
    #         neg       0.60      0.53      0.56        75
    #         pos       0.58      0.64      0.61        76
    #
    # avg / total       0.59      0.59      0.59       151
    #
    # Results for LinearSVC()
    # Training time: 0.054336s; Prediction time: 0.000749s
    #              precision    recall  f1-score   support
    #
    #         neg       0.59      0.49      0.54        75
    #         pos       0.57      0.66      0.61        76
    #
    # avg / total       0.58      0.58      0.57       151
