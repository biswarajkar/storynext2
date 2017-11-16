# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import glob
from nltk.tokenize import word_tokenize
from random import randint
from nltk.corpus import stopwords
import sys

# Create a class document
class document:
    def __init__(self):
        self.words = []
        self.classLabel = None
        self.wordCount = 0
        
    def addWord(self, word):
        self.words.append(word)
        self.wordCount = self.wordCount + 1
        
    def addLabel(self, docClass):
        self.classLabel = docClass


# Load the data
def load_data(fileslist, data_dict, label):
    tmp_vocab = set()
    global pos_words_list
    global neg_words_list
    
    for i in range(len(fileslist)):
        files = open(fileslist[i], 'r')
        lines = files.read().splitlines()
        
        wordlist = []
        for j in range(len(lines)):
            wordlist+= word_tokenize(lines[j])
        
        if label == 'pos':
            pos_words_list+= wordlist
        else:
            neg_words_list+= wordlist
            
        a = document()
        a.addLabel(label)
        for word in wordlist:
            a.addWord(word)
            tmp_vocab.add(word)
        
        data_dict[len(data_dict)] = a

    return(tmp_vocab)
    
    
# Convert words in every review to ids
def calc_word_ids(data):
    ids = []
    for i in range(len(data)):
        review = []
        for word in data[i].words:
        #for word in sent:
            if word not in stop:
                try:
                    review.append(wordsList.index(word))
                except ValueError:
                    review.append(399999)
        ids.append(review)
        
    return(ids)

    
# Get train data of required batchsize and convert them to vectors
def getTrainBatch(batchSize):
    global ids_train    
    labels = []
    traindata = []

    for i in range(batchSize):
        if (i % 2 == 0):
            num = randint(0,699)
            labels.append([1,0])
            tmp = np.zeros(50)
            for j in range(len(ids_train[num])):
                tmp+=wordVectors[ids_train[num][j]]
            traindata.append(tmp/len(ids_train[num]))
        else:
            num = randint(700,1399)
            labels.append([0,1])
            tmp = np.zeros(50)
            for j in range(len(ids_train[num])):
                tmp+=wordVectors[ids_train[num][j]]
            traindata.append(tmp/len(ids_train[num]))
    
    return (traindata, labels)

    
# Get entire test data and convert them to vectors
def getData():
    labels = []
    testdata = []
    for i in range(len(ids_test)):
        if (i<300):
            labels.append([1,0])
            tmp = np.zeros(50)
            for j in range(len(ids_test[i])):
                tmp+=wordVectors[ids_test[i][j]]
            testdata.append(tmp/len(ids_test[i]))
        else:
            labels.append([0,1])
            tmp = np.zeros(50)
            for j in range(len(ids_test[i])):
                tmp+=wordVectors[ids_test[i][j]]
            testdata.append(tmp/len(ids_test[i]))
            
    return (testdata, labels)
    
# Train the multilayer perceptron model - version 1
def multilayer_perceptron(n_input, n_hidden, n_output, learning_rate, epochs, batchSize, filepath, predictions_file):
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    # weights
    W1 = tf.Variable(tf.random_uniform([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_uniform([n_hidden, n_output], -1.0, 1.0))
    
    # bias
    b1 = tf.Variable(tf.random_uniform([n_hidden]), name = "Bias1")
    b2 = tf.Variable(tf.random_uniform([n_output]), name = "Bias2")
    
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
    hy = tf.sigmoid(tf.matmul(L2, W2) + b2)
    
    cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as session:
        session.run(init)
        
        for step in range(epochs):
            x_data, y_data = getTrainBatch(batchSize)
            session.run(optimizer, feed_dict={X: x_data, Y: y_data})
            
            #if step % 100 ==0:
            #    print(session.run(cost, feed_dict={X: x_data, Y: y_data}), step)
                
            #if (step % 1000 == 0 and step != 0):
            #    save_path = saver.save(session, filepath, global_step=step)
            #    print("saved to %s" % save_path)
                
        #save_path = saver.save(session, filepath, global_step=step)
        #print("saved to %s" % save_path)
        #answer = tf.equal(tf.floor(hy + 0.5), Y)
        #answer = tf.floor(hy + 0.5)
        answer = tf.equal(tf.argmax(hy,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(answer, "float"))
        
        #print(session.run([hy], feed_dict={X: x_data, Y: y_data}))
        print("Accuracy on train set: ", accuracy.eval({X: x_data, Y: y_data})* 100, "%")
        
        nextBatch, nextBatchLabels = getData()
        pred = session.run([hy], feed_dict={X: nextBatch})
        print("Accuracy for test set:", (session.run(accuracy, {X: nextBatch, Y: nextBatchLabels})) * 100)
        
    #with tf.Session() as session:
        #session.run(init)
        #saver.restore(session, save_path)
        
        #nextBatch, nextBatchLabels = getData()
        #pred = session.run([hy], feed_dict={X: nextBatch})
        #print("Accuracy for test set:", (session.run(accuracy, {X: nextBatch, Y: nextBatchLabels})) * 100)
    
    pred_file = open(predictions_file, 'a')
    
    for i in range(len(pred[0])):
        if pred[0][i][0]>pred[0][i][1]:
            pred_file.writelines('pos\n')
        else:
            pred_file.writelines('neg\n')
        
    pred_file.close()
    
    
# Train the multilayer perceptron model - version 2
def multilayer_perceptron_v2(n_input, n_hidden, n_output, learning_rate, epochs, batchSize, filepath, predictions_file):
    X = tf.placeholder(tf.float32)
    Y = tf.placeholder(tf.float32)
    
    # weights
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden], -1.0, 1.0))
    W2 = tf.Variable(tf.random_normal([n_hidden, n_hidden], -1.0, 1.0))
    W3 = tf.Variable(tf.random_normal([n_hidden, n_output], -1.0, 1.0))
    
    # bias
    b1 = tf.Variable(tf.random_normal([n_hidden], -1, 1), name = "Bias1")
    b2 = tf.Variable(tf.random_normal([n_hidden], -1, 1), name = "Bias2")
    b3 = tf.Variable(tf.random_normal([n_output], -1, 1), name = "Bias3")
    
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)
    L3 = tf.sigmoid(tf.matmul(L2, W2) + b2)
    hy = tf.sigmoid(tf.matmul(L3, W3) + b3)
    
    cost = tf.reduce_mean(-Y*tf.log(hy) - (1-Y)*tf.log(1-hy))
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    #init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    with tf.Session() as session:
        session.run(init)
        
        for step in range(epochs):
            x_data, y_data = getTrainBatch(batchSize)
            session.run(optimizer, feed_dict={X: x_data, Y: y_data})
            
            #if step % 100 ==0:
            #    print(session.run(cost, feed_dict={X: x_data, Y: y_data}), step)
                
            #if (step % 1000 == 0 and step != 0):
            #    save_path = saver.save(session, filepath, global_step=step)
            #    print("saved to %s" % save_path)
                
        #save_path = saver.save(session, filepath, global_step=step)
        #print("saved to %s" % save_path)
        #answer = tf.equal(tf.floor(hy + 0.5), Y)
        #answer = tf.floor(hy + 0.5)
        answer = tf.equal(tf.argmax(hy,1), tf.argmax(Y,1))
        accuracy = tf.reduce_mean(tf.cast(answer, "float"))
        
        #print(session.run([hy], feed_dict={X: x_data, Y: y_data}))
        print("Accuracy on train set: ", accuracy.eval({X: x_data, Y: y_data})* 100, "%")
        
        nextBatch, nextBatchLabels = getData()
        pred = session.run([hy], feed_dict={X: nextBatch})
        print("Accuracy for test set:", (session.run(accuracy, {X: nextBatch, Y: nextBatchLabels})) * 100)
        
    #with tf.Session() as session:
    #    session.run(init)
    #    saver.restore(session, save_path)
        
    #    nextBatch, nextBatchLabels = getData()
    #    pred = session.run([hy], feed_dict={X: nextBatch})
    #    print("Accuracy for test set:", (session.run(accuracy, {X: nextBatch, Y: nextBatchLabels})) * 100)
    
    pred_file = open(predictions_file, 'a')
        
    for i in range(len(pred[0])):
        if pred[0][i][0]>pred[0][i][1]:
            pred_file.writelines('pos\n')
        else:
            pred_file.writelines('neg\n')
        
    pred_file.close()
    
# Calculate precision, recall, f1 scores
def calc_scores_test(pred):
    nextBatch, nextBatchLabels = getData()
    
    pos_tp = 0
    pos_fp = 0
    pos_tn = 0
    pos_fn = 0
    pred_act = []
    
    for i in range(len(pred[0])):
        if pred[0][i][0]>pred[0][i][1]:
            pred_act.append([1,0])
        else:
            pred_act.append([0,1])
    
    for i in range(len(pred_act)):
        if pred_act[i][0] == 1 and nextBatchLabels[i][0] == 1:
            pos_tp+=1
        if pred_act[i][0] == 0 and nextBatchLabels[i][0] == 1:
            pos_fp+=1
        if pred_act[i][0] == 0 and nextBatchLabels[i][0] == 0:
            pos_tn+=1
        if pred_act[i][0] == 1 and nextBatchLabels[i][0] == 0:
            pos_fn+=1
    
    # For pos
    pos_precision = pos_tp/(pos_tp+pos_fp)
    pos_recall = pos_tp/(pos_tp+pos_fn)
    pos_f1 = 2 * pos_precision * pos_recall/(pos_precision + pos_recall)
    
    # For neg
    neg_precision = pos_tn/(pos_tn+pos_fn)
    neg_recall = pos_tn/(pos_tn+pos_fp)
    neg_f1 = 2 * neg_precision * neg_recall/(neg_precision + neg_recall)
        
    print("For pos - ")
    print("Precision: ", pos_precision)
    print("Recall: ", pos_recall)
    print("F1: ", pos_f1)
    
    print("For neg - ")
    print("Precision: ", neg_precision)
    print("Recall: ", neg_recall)
    print("F1: ", neg_f1)
################################################################
inputdata = sys.argv
train_folder = inputdata[1]
test_folder = inputdata[2]
wordlistfile = inputdata[3]
wordvectorfile = inputdata[4]
pred1 = inputdata[5]
pred2 = inputdata[6]
pred3 = inputdata[7]
pred4 = inputdata[8]

#base = 'C:/Users/Krishna/Desktop/Data Science/Northeastern University/NEU/Study Material/CS 6120 - Natural Language Processing/Assignments/HW2/hw2-sa-ds/'
stop = set(stopwords.words('english'))

ip_pos = glob.glob(train_folder + '/pos/*')
ip_neg = glob.glob(train_folder + '/neg/*')
train_data = dict()
vocab_train = set()
pos_words_list = []
neg_words_list = []

vocab_train = vocab_train.union(load_data(ip_pos, train_data, 'pos'))
vocab_train = vocab_train.union(load_data(ip_neg, train_data, 'neg'))

test_data = dict()
vocab_test = set()
ip_pos = glob.glob(test_folder + '/pos/*')
ip_neg = glob.glob(test_folder + '/neg/*')

vocab_test = vocab_test.union(load_data(ip_pos, test_data, 'pos'))
vocab_test = vocab_test.union(load_data(ip_neg, test_data, 'neg'))


wordsList = np.load(wordlistfile)
print('Loaded the word list..')
wordsList = wordsList.tolist() #Originally loaded as numpy array
wordsList = [word.decode('UTF-8') for word in wordsList] #Encode words as UTF-8
wordVectors = np.load(wordvectorfile)
print ('Loaded the word vectors..')

print("Calculating word-ids for all reviews to convert words to vectors..")
ids_train = calc_word_ids(train_data)
ids_test = calc_word_ids(test_data)

################################################################
#chkpt_flder = 'C:\\Users\\Krishna\\Desktop\\Data Science\\Northeastern University\\NEU\\Study Material\\CS 6120 - Natural Language Processing\\Assignments\\HW2\\sentiment analysis example\\training_data\\models\\1\\'
print("Training perceptron with a single hidden layers containing 25 hidden nodes..")
pred = multilayer_perceptron(n_input = 50,
                      n_hidden = 25,
                      n_output = 2,
                      learning_rate = 0.001,
                      epochs = 10000,
                      batchSize = 50,
                      filepath = "mlp1.ckpt",
                      predictions_file = pred1)

print("\nTraining perceptron with a single hidden layers containing 50 hidden nodes..")
pred = multilayer_perceptron(n_input = 50,
                      n_hidden = 50,
                      n_output = 2,
                      learning_rate = 0.001,
                      epochs = 10000,
                      batchSize = 50,
                      filepath = "mlp2.ckpt",
                      predictions_file = pred2)

print("\nTraining perceptron with 2 hidden layers and 25 hidden nodes each..")
pred = multilayer_perceptron_v2(n_input = 50,
                      n_hidden = 25,
                      n_output = 2,
                      learning_rate = 0.001,
                      epochs = 10000,
                      batchSize = 50,
                      filepath = "mlp3.ckpt",
                      predictions_file = pred3)

print("\nTraining perceptron with 2 hidden layers and 50 hidden nodes each..")
pred = multilayer_perceptron_v2(n_input = 50,
                      n_hidden = 50,
                      n_output = 2,
                      learning_rate = 0.001,
                      epochs = 10000,
                      batchSize = 50,
                      filepath = "mlp4.ckpt",
                      predictions_file = pred4)

###############################################################

