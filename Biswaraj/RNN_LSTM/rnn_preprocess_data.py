import numpy as np, time
from datetime import datetime

start_time = time.time()
print("\n", str(datetime.now()), " - Load Word Lists and GloVe Vectors - ")

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

# Process the Training data (IMDB 25k movie reviews, 12.5k pos : 12.5 neg)
from os import listdir
from os.path import isfile, join

# Create a list of Positive and Negative File names
positiveFiles = ['positiveReviews/' + f for f in listdir('positiveReviews/') if isfile(join('positiveReviews/', f))]
negativeFiles = ['negativeReviews/' + f for f in listdir('negativeReviews/') if isfile(join('negativeReviews/', f))]

# ------------------------------
# Get Training Corpus Statistics
# ------------------------------
numWords = []
print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Pre-processing Input reviews!")

# Read each file from the Positive Reviews Folder
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
# Read each file from the Negative Reviews Folder
for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numWords.append(counter)
numFiles = len(numWords)
print('The total number of files is', numFiles)
print('The total number of words in the files is', sum(numWords))

# Most Reviews have less than 250 words, put an upper bound
maxReviewWordLength = 250

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " -- Pre-Processing Training data finished --", " (", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ")")

