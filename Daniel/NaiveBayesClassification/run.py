import glob

from NaiveBayesClassification.training import training
from NaiveBayesClassification.training import naive_classification

# training
# classes store words totally for each class
# docs store number of files, should mannually input
classes = {'neg': 0, 'pos': 0}
docs = {'neg': 700, 'pos': 700}
training(classes, docs)

# test files
folder = '../Dataset/test/pos/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == naive_classification(data):
            pos += 1
        else:
            neg += 1
print(pos, neg)

# test gutenberg - positive
folder = '../Dataset/test/gutenberg/pos/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == naive_classification(data):
            pos += 1
        else:
            neg += 1
print()
print("Gutenberg POS --> [num guessed pos, num guessed neg]")
print(pos, neg)

# test gutenberg - negative
folder = '../Dataset/test/gutenberg/neg/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == naive_classification(data):
            pos += 1
        else:
            neg += 1
print()
print("Gutenberg NEG --> [num guessed pos, num guessed neg]")
print(pos, neg)


# test news - positive
folder = '../Dataset/test/news/pos/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == naive_classification(data):
            pos += 1
        else:
            neg += 1
print()
print("News POS --> [TP, FN]")
print(pos, neg)

# test news - negative
folder = '../Dataset/test/news/neg/*.txt'
files = glob.glob(folder)
pos = 0;
neg = 0
for fle in files:
    # open the file and then call .read() to get the text
    with open(fle, "r", encoding='utf-8', errors='ignore') as f:
        data = f.read()
        if 'pos' == naive_classification(data):
            pos += 1
        else:
            neg += 1
print()
print("News NEG --> [FP, TN]")
print(pos, neg)
