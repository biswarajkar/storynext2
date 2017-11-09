import numpy as np
import time
from datetime import datetime

start_time = time.time()
print("\n", str(datetime.now()), " - Start Test Data Processing - ")

# Define a function to pre-process data and get data statistics
# Load a list of 4,00,000 words from the GloVe Dataset (as a numpy array)
wordsList = np.load('wordsList.npy')
print('Loaded the word list!')
# Convert the wordList from numpy array to a list
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
# Load the word vectors for the words (50 dimension vector for each word)
wordVectors = np.load('wordVectors.npy')
print('Loaded the word vectors!')

# Process the Test data
from os import listdir
from os.path import isfile, join

# Create a list of Positive and Negative File names
positiveTestFiles = ['test/positive/' + f for f in listdir('test/positive/') if isfile(join('test/positive/', f))]
negativeTestFiles = ['test/negative/' + f for f in listdir('test/negative/') if isfile(join('test/negative/', f))]

# ------------------------------
# Get Test Corpus Statistics
# ------------------------------
numTestWords = []
print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Pre-processing Test Articles!")

# Read each file from the Positive articles Folder
for pf in positiveTestFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numTestWords.append(counter)
# Read each file from the Negative articles Folder
for nf in negativeTestFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numTestWords.append(counter)
numTestFiles = len(numTestWords)
print('The total number of Test files is', numTestFiles)
print('The total number of words in the Test files is', sum(numTestWords))


# Define a function to remove all special characters
def cleanSentences(string):
    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


start_time = time.time()
print("\n - Vectorizing all Test Articles Started - ")

# Most articles have less than 250 words, put an upper bound
maxArticleWordLength = 250

# ------------------------------
#  Vectorize Test Corpus
# ------------------------------
# Create an empty numpy array to store the vectorized representation of the articles
test_ids = np.zeros((numTestFiles, maxArticleWordLength), dtype='int32')
fileCounter = 0
true_test_labels = np.zeros(numTestFiles, dtype='int32')

# Read each of the Positive article (File) and vectorize the files into a matrix
for pf in positiveTestFiles:
    with open(pf, "r") as f:
        true_test_labels[fileCounter] = 1
        testWordIndexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                test_ids[fileCounter][testWordIndexCounter] = wordsList.index(word)
            except ValueError:
                test_ids[fileCounter][testWordIndexCounter] = 399999  # Vector for unknown words
            testWordIndexCounter = testWordIndexCounter + 1
            # If we see articles with more than 250 words, we discard the excess words
            if testWordIndexCounter >= maxArticleWordLength:
                break
        fileCounter = fileCounter + 1

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Positive Test files vectorization finished!")

# Read each of the Negative article (File) and vectorize the files into a matrix
for nf in negativeTestFiles:
    with open(nf, "r") as f:
        true_test_labels[fileCounter] = 0
        testWordIndexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                test_ids[fileCounter][testWordIndexCounter] = wordsList.index(word)
            except ValueError:
                test_ids[fileCounter][testWordIndexCounter] = 399999  # Vector for unknown words
            testWordIndexCounter = testWordIndexCounter + 1
            # If we see articles with more than 250 words, we discard the excess words
            if testWordIndexCounter >= maxArticleWordLength:
                break
        fileCounter = fileCounter + 1

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Negative Test files vectorization finished!")

# Save the Matrix representing all test files
np.save('vectorizedTestMatrix', test_ids)
# Save the Matrix representing all test file labels (true labels)
np.save('trueTestLabels', true_test_labels)

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), "-- Vectorized Test Saved --", " (", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ")")

