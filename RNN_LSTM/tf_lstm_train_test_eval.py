
from rnn_preprocess_data import wordVectors
from rnn_vectorize_training import maxReviewWordLength
from tf_support_functions import getTrainBatch, getTestBatch, getCustomTestBatch
import time, os
from datetime import datetime as dt
import matplotlib
matplotlib.use('TkAgg')
from pylab import *

# Capture Program Start Time
start_time = time.time()

# Code to capture all console output to a file
from capture_run import Tee
f = open('logs/'+os.path.basename(sys.argv[0]).strip(".py")+'_'+time.strftime("%Y%m%d-%H%M%S")+'_out.txt', 'w')
original = sys.stdout
sys.stdout = Tee(sys.stdout, f)

# Parameter Definitions
lstmUnits = 64           # Number of LSTM Units
numClasses = 2           # Number of Output Classes
iterations = 70000       # Number of Iterations
numDimensions = 300      # Dimensions for each word vector
learn_rate = 0.1         # Learning Rate for Optimizer
batchSize = 24           # Batch Size Training/Test
no_of_batches = 10       # No. of Test Batches
opt_lbl='GRDSC'          # Optimizer Name Abbr.
polling_interval = 1000  # Polling interval to record checkpoints during Training


print("\n", str(dt.now()), " - Started Building LSTM RNN Model - ")
import tensorflow as tf
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
# Value will have the last Hidden state vector
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
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learn_rate).minimize(loss)

print(time.strftime("%M:%S", time.gmtime(time.time() - start_time)), " Started Training the model")
print("### Batch Size:", batchSize, "| LSTM Units:", lstmUnits, "| Iterations:", iterations,
      "| Dimensions:", numDimensions, "| MaxWords:", maxReviewWordLength, "| Optimizer:", opt_lbl, learn_rate, "###")


# Lines 73-123 :Commented to use existing Trained model, Uncomment to train the model fresh
# WARNING: Training takes 7-8 hours for about 80000 iterations

# # ------------------------------
# # Train TF Model
# # ------------------------------
# sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# loss_array = np.zeros(iterations, dtype='int32')
#
# tf.summary.scalar('Loss', loss)
# tf.summary.scalar('Accuracy', accuracy)
# merged = tf.summary.merge_all()
# logdir = "tensorboard/"
# writer = tf.summary.FileWriter(logdir, sess.graph)
#
# for i in range(iterations):
#     # Next Batch of reviews
#     nextBatch, nextBatchLabels = getTrainBatch(batchSize, maxReviewWordLength)
#     _, loss_val = sess.run([optimizer, loss], {input_data: nextBatch, labels: nextBatchLabels})
#     loss_array[i] = loss_val*100
#
#     # Write summary to Tensorboard
#     if (i % (polling_interval/10) == 0 and i != 0):
#         summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
#         writer.add_summary(summary, i)
#
#     # Save the network every 'polling_interval' number of training iterations
#     if (i % polling_interval == 0 and i != 0):
#
#         # Save state
#         save_path = saver.save(sess, "models/trained_lstm.ckpt", global_step=i)
#         print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), "Saved to %s" % save_path, "| Loss: ", loss_val)
#
# # Save the Loss Values
# np.save('processing/loss_array_NLS' + lstmUnits.__str__() + 'ITR' + iterations.__str__()
#         + 'MAXW' + maxReviewWordLength.__str__() + 'OPT' + opt_lbl + learn_rate.__str__(), loss_array)
#
# # Plot the Loss Function
# loss_array = np.load('processing/loss_array_NLS' + lstmUnits.__str__() + 'ITR' + iterations.__str__()
#         + 'MAXW' + maxReviewWordLength.__str__() + 'OPT' + opt_lbl + learn_rate.__str__()+'.npy')
# import matplotlib.pyplot as plt_loss
# plt_loss.plot(arange(0.0, iterations, 1), loss_array.tolist())
# plt_loss.xlabel('Number of Iterations')
# plt_loss.ylabel('Loss')
# plt_loss.title('Loss Variations: LSTMs ' + lstmUnits.__str__() + ' | MaxWords ' + maxReviewWordLength.__str__()
#                + ' | Optimizer ' + opt_lbl + learn_rate.__str__())
# plt_loss.grid(True)
# plt_loss.savefig('plots/Loss_NLS' + lstmUnits.__str__() + 'ITR' + iterations.__str__()
#                  + 'MAXW' + maxReviewWordLength.__str__() + 'OPT' + opt_lbl + learn_rate.__str__() + '.png', bbox_inches='tight', dpi=500)
# plt_loss.close()
#
# print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), " Completed Training the model")

# ------------------------------
# Run TF Model on Test
# ------------------------------

# 2 Sessions: current session and saved model session
saver = tf.train.Saver()
sess = tf.InteractiveSession()
saver.restore(sess, tf.train.latest_checkpoint('models'))


print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), " Running Model on partitioned test data")
test_accuracy_array=[]
for i in range(no_of_batches):
    nextBatch, nextBatchLabels = getTestBatch(batchSize, maxReviewWordLength)
    t_acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
    test_accuracy_array.append(t_acc)
    print("Accuracy for this Test batch:", t_acc * 100)

# Vectorize Hand-tagged Test Data
from rnn_vectorize_test import vectorizeTest
vectorizeTest()

print("\n", time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), " Running Model on hand tagged data")
handtag_accuracy_array=[]
for i in range(no_of_batches):
    nextBatch, nextBatchLabels = getCustomTestBatch(batchSize, maxReviewWordLength)
    h_acc = sess.run(accuracy, {input_data: nextBatch, labels: nextBatchLabels})
    handtag_accuracy_array.append(h_acc)
    print("Accuracy for this Custom Test batch:", h_acc * 100)


# Plot the Accuracy Function
import matplotlib.pyplot as plt_accu
plt_accu.plot(arange(0.0, no_of_batches, 1), test_accuracy_array)
plt_accu.plot(arange(0.0, no_of_batches, 1), handtag_accuracy_array, linestyle="--")
plt_accu.xlabel('Number of Batches')
plt_accu.ylabel('Accuracy')
plt_accu.title('Accuracy Variations: LSTMs ' + lstmUnits.__str__() + ' | Iterations ' + iterations.__str__()
               + ' | MaxWords ' + maxReviewWordLength.__str__() + ' | Optimizer ' + opt_lbl + learn_rate.__str__())
plt_accu.grid(True)
plt_accu.savefig('plots/Accuracy_NLS' + lstmUnits.__str__() + 'ITR' + iterations.__str__()
                 + 'MAXW' + maxReviewWordLength.__str__() + 'OPT' + opt_lbl + learn_rate.__str__() + '.png', bbox_inches='tight', dpi=300)
plt_accu.close()

print(time.strftime("%H:%M:%S", time.gmtime(time.time() - start_time)), "-- Evaluation Completed --",
      " (", dt.now().strftime('%Y-%m-%d %H:%M:%S'), ")")