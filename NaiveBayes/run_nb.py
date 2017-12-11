import glob
import sys
from train_nb import training, classifier_nb


print("Running Naive Bayes using 25K reviews IMDB corpus and redirecting output to 'results.out'")

f = open("results.out", 'w')
sys.stdout = f

# Define Training parameters
classes = {'negative': 0, 'positive': 1}
docs = {'neg': 12500, 'pos': 12500}

# Train model
training(classes, docs)

# Run on Test Data
tp = 0
fn = 0
fp = 0
tn = 0

# Test Hand Tagged Data - Positive
folder = '../testing_corpus_hand_tagged/positive/*.txt'
files = glob.glob(folder)
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == classifier_nb(data):
            tp += 1
        else:
            fn += 1
print()
print("Positive Classifier --> [TP, FN]")
print(tp, fn)

# Test Hand Tagged Data - Negative
folder = '../testing_corpus_hand_tagged/negative/*.txt'
files = glob.glob(folder)
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == classifier_nb(data):
            fp += 1
        else:
            tn += 1
print()
print("Negative Classification --> [FP, TN]")
print(fp, tn)

# Report Metrics
precision_p = tp / (tp + fp)
recall_p = tp / (tp + fn)
f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
acc_p = tp / (tp + fp)
print("\nPositive Reviews:")
print("   Precision:", precision_p)
print("   Recall:", recall_p)
print("   F1:", f1_p)
print("   Accuracy:", acc_p)

precision_n = tn / (tn + fn)
recall_n = tn / (tn + fp)
f1_n = (2 * precision_n * recall_n) / (precision_n + recall_n)
acc_n = tn / (tn + fn)
print("\nNegative Reviews:")
print("   Precision:", precision_n)
print("   Recall:", recall_n)
print("   F1:", f1_n)
print("   Accuracy:", acc_n)

f.close()
