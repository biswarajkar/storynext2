
from rnn_preprocess_data import wordVectors, maxReviewWordLength
from tf_support_functions import getTrainBatch, getTestBatch, getCustomTestBatch
import time
from datetime import datetime

start_time = time.time()

batchSize = 24
lstmUnits = 64
numClasses = 2  # Number of output classes
iterations = 100000 # Number of iterations
numDimensions = 300  # Dimensions for each word vector

print("\n", str(datetime.now()), " - Started Building LSTM RNN Model - ")
import tensorflow as tf
with tf.Session() as sess:
    print("")

tf.reset_default_graph()

# Define placeholders for the Class labels and the input data
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.int32, [batchSize, maxReviewWordLength])

# ------------------------------
# Build the TF Model
# ------------------------------

data = tf.Variable(tf.zeros([batchSize, maxReviewWordLength, numDimensions]), dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors, input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)

# RNN Processing
# value will have the last Hidden state vector
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

# Calculate Accuracy and Loss
correctPred = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)


# print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Started Training the model")
# # ------------------------------
# # Train TF Model
# # ------------------------------
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
#
# for i in range(iterations):
#     # Next Batch of reviews
#     nextBatch, nextBatchLabels = getTrainBatch(batchSize);
#     sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
#
#     # Save the network every 1,000 training iterations
#     if (i % 1000 == 0 and i != 0):
#         save_path = saver.save(sess, "models/trained_lstm.ckpt", global_step=i)
#         print(time.strftime("%H%M:%S", time.gmtime(time.time() - start_time)), "Saved to %s" % save_path)
#
# print(time.strftime("%H%M:%S", time.gmtime(time.time() - start_time)), " Completed Training the model")

# ------------------------------
# Run TF Model on Test
# ------------------------------
# 2 Sessions: current session and saved model session
saver = tf.train.Saver()
sess = tf.InteractiveSession()
# saver = tf.train.import_meta_graph('models/pretrained_lstm.ckpt-90000.meta')
# saver.restore(sess, 'models/pretrained_lstm.ckpt-90000')
saver.restore(sess, tf.train.latest_checkpoint('models'))

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Running Model on test data")
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getTestBatch(batchSize)
    print("Accuracy for this Test batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)


print("\n", time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Running Model on custom tagged data")
iterations = 10
for i in range(iterations):
    nextBatch, nextBatchLabels = getCustomTestBatch(batchSize)
    print("Accuracy for this Custom Test batch:", (sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})) * 100)


print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), "-- Evaluation Completed --", " (", datetime.now().strftime('%Y-%m-%d %H:%M:%S'), ")")
