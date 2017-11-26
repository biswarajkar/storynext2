import numpy as np
import time
from datetime import datetime

from rnn_preprocess_data import numTrainingFiles, positiveFiles, negativeFiles, wordsList, maxReviewWordLength

# Define a function to remove all special characters
def cleanSentences(string):
    import re
    strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

start_time = time.time()
print("\n", str(datetime.now()), " - Vectorizing all 25K Input Reviews Started - ")

# ------------------------------
#  Vectorize Training Corpus
# ------------------------------
# Create an empty numpy array to store the vectorized representation of the 25k reviews
ids = np.zeros((numTrainingFiles, maxReviewWordLength), dtype='int32')
fileCounter = 0

# Read each of the 12.5k Positive reviews (File) and vectorize the files into a matrix
for pf in positiveFiles:
    with open(pf, "r") as f:
        wordIndexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][wordIndexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][wordIndexCounter] = 399999  # Vector for unknown words
            wordIndexCounter = wordIndexCounter + 1
            # If we see reviews with more than 250 words, we discard the excess words
            if wordIndexCounter >= maxReviewWordLength:
                break
        fileCounter = fileCounter + 1

print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), " Positive files vectorization finished!")

# Read each of the 12.5k Negative review (File) and vectorize the files into a matrix
for nf in negativeFiles:
    with open(nf, "r") as f:
        wordIndexCounter = 0
        line = f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][wordIndexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][wordIndexCounter] = 399999  # Vector for unknown words
            wordIndexCounter = wordIndexCounter + 1
            # If we see reviews with more than 250 words, we discard the excess words
            if wordIndexCounter >= maxReviewWordLength:
                break
        fileCounter = fileCounter + 1

print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), " Negative files vectorization finished!")

# Save the Matrix representing all the 25k reviews
np.save('vectorizedTrainingMatrix', ids)

print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), "-- Vectorized Reviews Saved --", " (", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ")")

