import numpy as np, time
from datetime import datetime as dt
import matplotlib
matplotlib.use('TkAgg')

start_time = time.time()
print("\n", str(dt.now()), " - Load Word Lists and GloVe Vectors - ")

# Define a function to pre-process data and get data statistics
# Load a list of 4,00,000 words from the GloVe Dataset (as a numpy array)
wordsList = np.load('processing/wordsList.npy')
print('Loaded the word list!')
# Convert the wordList from numpy array to a list
wordsList = wordsList.tolist()
wordsList = [word.decode('UTF-8') for word in wordsList]  # Encode words as UTF-8
# Load the word vectors for the words (50 dimension vector for each word)
wordVectors = np.load('processing/wordVectors.npy')
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
numTrainingWords = []
print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Pre-processing Input reviews!")

# Read each file from the Positive Reviews Folder
for pf in positiveFiles:
    with open(pf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numTrainingWords.append(counter)
# Read each file from the Negative Reviews Folder
for nf in negativeFiles:
    with open(nf, "r", encoding='utf-8') as f:
        line = f.readline()
        counter = len(line.split())
        numTrainingWords.append(counter)
numTrainingFiles = len(numTrainingWords)
print('The total number of files is', numTrainingFiles)
print('The total number of words in the files is', sum(numTrainingWords))

# Plot the Word Distribution
import matplotlib.pyplot as plt_tr
plt_tr.hist(numTrainingWords, 50)
plt_tr.xlabel('No of words on File')
plt_tr.ylabel('Frequency')
plt_tr.title('Frequency Distribution of ' + sum(numTrainingWords).__str__()
             + ' words in ' + numTrainingFiles.__str__() + ' Training Files')
plt_tr.savefig('plots/TrainingDataWordDistribution.png')
plt_tr.close()

# Most Reviews have less than 300 words, put an upper bound
maxReviewWordLength = 300

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " -- Pre-Processing Training data finished --", " (", dt.now().strftime('%Y-%m-%d %H:%M:%S'), ")")

