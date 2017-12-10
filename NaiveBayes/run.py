import glob
from training import training, classifier_nb

import sys
f = open("results.out", 'w')
sys.stdout = f

# Define Training parameters
classes = {'neg': 0, 'pos': 1}
docs = {'neg': 700, 'pos': 700}

# Train model
training(classes, docs)

# Run on Test Data
tp = 0
fn = 0
fp = 0
tn = 0

# Test Gutenberg - positive
folder = 'data/test/gutenberg/pos/*.txt'
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
print("Gutenberg POS --> [num guessed pos, num guessed neg]")
print(tp, fn)


# Test Gutenberg - negative
folder = 'data/test/gutenberg/neg/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == classifier_nb(data):
            fp += 1
        else:
            tn += 1
print()
print("Gutenberg NEG --> [num guessed pos, num guessed neg]")
print(fp, tn)

precision_p = tp/(tp+fp)
recall_p = tp/(tp+fn)
f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
print("\nPositive Reviews:")
print("   Precision:", precision_p)
print("   Recall:", recall_p)
print("   F1:", f1_p)

# Reset Metrics
tp=0
fn=0
fp=0
tn = 0
precision_p=0
recall_p=0
f1_p = 0

# Test News - Positive
folder = 'data/test/news/pos/*.txt'
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
print("News POS --> [TP, FN]")
print(tp, fn)

# Test News - Negative
folder = 'data/test/news/neg/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == classifier_nb(data):
            fp += 1
        else:
            tn += 1
print()
print("News NEG --> [FP, TN]")
print(fp, tn)

# Report Metrics
precision_p = tp/(tp+fp)
recall_p = tp/(tp+fn)
f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
print("\nPositive Reviews:")
print("   Precision:", precision_p)
print("   Recall:", recall_p)
print("   F1:", f1_p)

# Reset Metrics

tp = 0
fn = 0
fp = 0
tn = 0

# Test All - positive
folder = 'data/test/pos/*.txt'
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
print("All POS --> [num guessed pos, num guessed neg]")
print(tp, fn)


# Test All - negative
folder = 'data/test/neg/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == classifier_nb(data):
            fp += 1
        else:
            tn += 1
print()
print("All NEG --> [num guessed pos, num guessed neg]")
print(fp, tn)

# Report Metrics
precision_p = tp/(tp+fp)
recall_p = tp/(tp+fn)
f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
print("\nPositive Reviews:")
print("   Precision:", precision_p)
print("   Recall:", recall_p)
print("   F1:", f1_p)

precision_n = tn/(tn+fn)
recall_n = tn/(tn+fp)
f1_n = (2 * precision_n * recall_n) / (precision_n + recall_n)
print("\nNegative Reviews:")
print("   Precision:", precision_n)
print("   Recall:", recall_n)
print("   F1:", f1_n)
f.close()
