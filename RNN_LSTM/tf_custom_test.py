import numpy as np
import tensorflow as tf


numDimensions = 300
maxSeqLength = 250
batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000


wordsList = np.load('wordsList.npy').tolist()
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load('wordVectors.npy')

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxSeqLength])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.25)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

sess = tf.InteractiveSession()
saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint('models'))

# Removes punctuation, parentheses, question marks, etc., and leaves only alphanumeric characters
import re
strip_special_chars = re.compile("[^A-Za-z0-9 ]+")

def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())

# Process the Test data
from os import listdir
from os.path import isfile, join

positiveTestFiles = ['test/positive/' + f
                     for f in listdir('test/positive/') if isfile(join('test/positive/', f))]
negativeTestFiles = ['test/negative/' + f
                     for f in listdir('test/negative/') if isfile(join('test/negative/', f))]

# Create an empty numpy array to store the vectorized representation of the articles
test_ids = np.zeros((36, maxSeqLength), dtype='int32')
fileCounter = 0
batchLabels=[]
tp=0
fp=0
tn=0
fn=0

# Read each of the Positive article (File) and vectorize the files into a matrix
for pf in positiveTestFiles:
    with open(pf, "r") as f:
        batchLabels.append([1, 0])
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
            if testWordIndexCounter >= maxSeqLength:
                break
        # predictedSentiment = sess.run(prediction, {input_data: test_ids[fileCounter]})[0]
        # if (predictedSentiment[0] > predictedSentiment[1]):
        #     tp += 1
        # else:
        #     fn += 1
        fileCounter = fileCounter + 1

predictedSentiment={}
# Read each of the Negative article (File) and vectorize the files into a matrix
for nf in negativeTestFiles:
    with open(nf, "r") as f:
        batchLabels.append([0, 1])
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
            if testWordIndexCounter >= maxSeqLength:
                break
        # predictedSentiment = sess.run(prediction, {input_data: test_ids[fileCounter]})[0]
        # if (predictedSentiment[1] > predictedSentiment[0]):
        #     tn += 1
        # else:
        #     fp += 1
        fileCounter = fileCounter + 1

print(batchLabels, len(batchLabels))

# Save the Matrix representing all test files
np.save('vectorizedTestMatrix', test_ids)

# # Read each of the Positive article (File) and vectorize the files into a matrix
# for pf in positiveTestFiles:
#     with open(pf, "r", encoding='utf-8') as f:
#         line = f.readline()
#         docMatrix = getSentenceMatrix(line)
#         predictedSentiment = sess.run(prediction, {input_data: docMatrix})[0]
#         (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100
#         if (predictedSentiment[0] > predictedSentiment[1]):
#             tp += 1
#         else:
#             fn += 1
#
# # Read each of the Negative article (File) and vectorize the files into a matrix
# for nf in negativeTestFiles:
#     with open(pf, "r", encoding='utf-8') as f:
#         line = f.readline()
#         docMatrix = getSentenceMatrix(line)
#         predictedSentiment = sess.run(prediction, {input_data: docMatrix})[0]
#         if (predictedSentiment[1] > predictedSentiment[0]):
#             tn += 1
#         else:
#             fp += 1

print(sess.run(accuracy, {input_data: test_ids, labels: batchLabels}))

print("TP, FN",tp,":",fn, "TN FP",tn,":",fp)
# Report Metrics
precision_p = tp/(tp+fp)
recall_p = tp/(tp+fn)
f1_p = (2 * precision_p * recall_p) / (precision_p + recall_p)
print("\nPositive Reviews:")
print("   Precision:", precision_p)
print("   Recall:", recall_p)
print("   F1:", f1_p)