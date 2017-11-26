from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from operator import add
from random import shuffle
import tensorflow as tf
import sklearn as sk
import sys
import os
from sklearn.metrics import confusion_matrix


glove={}
with open('dwr\\glove.6B.100d.txt',encoding='utf-8') as f:
    h=f.readlines()
    for line in h:
        array=line.strip().split(" ")
        word=array[0]
        vector=list(map(float,array[1:]))
        glove[word.lower()]=vector



#class={100:'neg',101:'pos'}
stop = set(stopwords.words('english'))
path='train\\neg'
x=[0]*100
batch=[]
for filename in os.listdir(path):
    with open(path+"\\"+filename) as f:
       lines=f.readlines()
       count=0
       for line in lines:
           for token in line.strip().split(" "):
               if token.lower() not in stop:
                   count+=1
                   try:
                       vector=glove[token.lower()]
                   except:
                       vector=[0]*100                   
                   x=list(map(add,x,vector))
    x=[i/count for i in x]
    x.extend([1,0])
    batch.append(x)
    x=[0]*100

path='train\\pos'
for filename in os.listdir(path):
    with open(path+"\\"+filename) as f:
       lines=f.readlines()
       count=0
       for line in lines:
           for token in line.strip().split(" "):
               if token.lower() not in stop:
                   count+=1
                   try:
                       vector=glove[token.lower()]
                   except:
                       vector=[0]*100                   
                   x=list(map(add,x,vector))                    
    x=[i/count for i in x]    
    x.extend([0,1])
    batch.append(x)
    x=[0]*100

shuffle(batch)
batch=np.array(batch)    
        
                                  
# Parameters
learning_rate = 0.001
training_epochs = 20000
display_step = 2
total_batch=1400


n_hidden_1 = 25 # Neurons in layer 1
n_hidden_2 = 25 # Neurons in layer 2
n_input = 100   #100 dimensions glove
n_classes = 2 # Positive and Negative

# tf Graph input
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_classes])

# weights & bias fro the hidden layers
w1=tf.Variable(tf.random_normal([n_input, n_hidden_1]))
w2=tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]))
w3=tf.Variable(tf.random_normal([n_hidden_2, n_classes]))


b1=tf.Variable(tf.random_normal([n_hidden_1]))
b2=tf.Variable(tf.random_normal([n_hidden_2]))
b3=tf.Variable(tf.random_normal([n_classes]))


path='test\\neg'
x=[0]*100
batch2=[]
for filename in os.listdir(path):
    with open(path+"\\"+filename) as f:
       lines=f.readlines()
       count=0
       for line in lines:
           for token in line.strip().split(" "):
               if token.lower() not in stop:
                   count+=1
                   try:
                       vector=glove[token.lower()]
                   except:
                       vector=[0]*100                   
                   x=list(map(add,x,vector))
    x=[i/count for i in x]
    x.extend([1,0])
    batch2.append(x)
    x=[0]*100

path='test\\pos'
for filename in os.listdir(path):
    with open(path+"\\"+filename) as f:
       lines=f.readlines()
       count=0
       for line in lines:
           for token in line.strip().split(" "):
               if token.lower() not in stop:
                   count+=1
                   try:
                       vector=glove[token.lower()]
                   except:
                       vector=[0]*100                   
                   x=list(map(add,x,vector))                    
    x=[i/count for i in x]    
    x.extend([0,1])
    batch2.append(x)
    x=[0]*100

shuffle(batch2)
batch2=np.array(batch2) 

print('one hidden layer, adam optimizer, 25 neurons, 20000 epoch')    
# One hidden layer and Adam optimizer
def multilayer_perceptron(x):

    layer_1 = tf.add(tf.matmul(x, w1), b1)
    out_layer = tf.matmul(layer_1, w3) + b3
    return out_layer

logits = multilayer_perceptron(X)

# loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# initializing variable
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    count=0
    for epoch in range(training_epochs):
        count+=1
        k=batch[np.random.choice(batch.shape[0], 200, replace=False), :]
        _, c = sess.run([train_op, loss_op], feed_dict={X: k[:,:-2],
                                                        Y: k[:,100:]})
        avg_cost = c / 200
        print("Epoch: "+str(epoch),"%.9f" % avg_cost)

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy1=accuracy.eval({X: batch2[:,:-2],Y: batch2[:,100:]})
    pred_y=sess.run(tf.argmax(pred, 1),{X: batch2[:,:-2],Y: batch2[:,100:]})
    true_y=np.argmax(batch2[:,100:],1)
    precision1=sk.metrics.precision_score(true_y,pred_y)
    recall1=sk.metrics.recall_score(true_y,pred_y)
    f1score1=sk.metrics.f1_score(true_y,pred_y)

print('for one hidden layer, gradient optimizer, 25 neurons, 20000 epoch')
# One hidden layer and gradient descent optimizer
def multilayer_perceptron(x):

    layer_1 = tf.add(tf.matmul(x, w1), b1)
    out_layer = tf.matmul(layer_1, w3) + b3
    return out_layer


logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        k=batch[np.random.choice(batch.shape[0], 200, replace=False), :]
        _, c = sess.run([train_op, loss_op], feed_dict={X: k[:,:-2],
                                                        Y: k[:,100:]})
        avg_cost = c / 200
        print("Epoch: "+str(epoch),"%.9f" % avg_cost)

    # Test model
    pred = tf.nn.softmax(logits)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy2=accuracy.eval({X: batch2[:,:-2],Y: batch2[:,100:]})
    pred_y=sess.run(tf.argmax(pred, 1),{X: batch2[:,:-2],Y: batch2[:,100:]})
    true_y=np.argmax(batch2[:,100:],1)
    precision2=sk.metrics.precision_score(true_y,pred_y)
    recall2=sk.metrics.recall_score(true_y,pred_y)
    f1score2=sk.metrics.f1_score(true_y,pred_y)

print('Two hidden layers, adam optimizer, 25 neurons, 20000 epoch')
#Added an extra hidden layer
def multilayer_perceptron(x):

    layer_1 = tf.add(tf.matmul(x, w1), b1)
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    out_layer = tf.matmul(layer_2, w3) + b3
    return out_layer
a=[1,2,3,4,5]
n=5
k=2 
b=[0,0,0,0,0]
while k<n:
    b[n-k-1]=a[n-1]
    n-=1
b=b[:-n]
b.extend(a[:n])
print(*b,sep=' ')
logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        k=batch[np.random.choice(batch.shape[0], 200, replace=False), :]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_op, loss_op], feed_dict={X: k[:,:-2],
                                                        Y: k[:,100:]})
        # Compute average loss
        avg_cost = c / 200
        print("Epoch: "+str(epoch),"%.9f" % avg_cost)

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy3=accuracy.eval({X: batch2[:,:-2],Y: batch2[:,100:]})
    pred_y=sess.run(tf.argmax(pred, 1),{X: batch2[:,:-2],Y: batch2[:,100:]})
    true_y=np.argmax(batch2[:,100:],1)
    precision3=sk.metrics.precision_score(true_y,pred_y)
    recall3=sk.metrics.recall_score(true_y,pred_y)
    f1score3=sk.metrics.f1_score(true_y,pred_y)



print('Two hidden layer, adam optimizer, 25 neurons, epoch 1000')
#increased the number of neurons
def multilayer_perceptron(x):

    layer_1 = tf.add(tf.matmul(x, w1), b1)
    layer_2 = tf.add(tf.matmul(layer_1, w2), b2)
    out_layer = tf.matmul(layer_2, w3) + b3
    return out_layer

logits = multilayer_perceptron(X)

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

init = tf.global_variables_initializer()
training_epochs=1000
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        k=batch[np.random.choice(batch.shape[0], 200, replace=False), :]
        # Run optimization op (backprop) and cost op (to get loss value)
        _, c = sess.run([train_op, loss_op], feed_dict={X: k[:,:-2],
                                                        Y: k[:,100:]})
        # Compute average loss
        avg_cost = c / 200
        print("Epoch: "+str(epoch),"%.9f" % avg_cost)

    # Test model
    pred = tf.nn.softmax(logits)  # Apply softmax to logits
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    accuracy4=accuracy.eval({X: batch2[:,:-2],Y: batch2[:,100:]})
    pred_y=sess.run(tf.argmax(pred, 1),{X: batch2[:,:-2],Y: batch2[:,100:]})
    true_y=np.argmax(batch2[:,100:],1)
    precision4=sk.metrics.precision_score(true_y,pred_y)
    recall4=sk.metrics.recall_score(true_y,pred_y)
    f1score4=sk.metrics.f1_score(true_y,pred_y)

print('for one hidden layer, adam optimizer, 25 neurons and 20000 epoch','Accuracy:'+str(accuracy1),'precision:'+str(precision1),'recall'+str(recall1),'f1score:'+str(f1score1))
print('for one hidden layer, gradient optimizer, 25 neurons and 20000 epoch','Accuracy:'+str(accuracy2),'precision:'+str(precision2),'recall'+str(recall2),'f1score:'+str(f1score2))
print('for two hidden layer, adam optimizer, 25 neurons and 20000 epoch','Accuracy:'+str(accuracy3),'precision:'+str(precision3),'recall'+str(recall3),'f1score:'+str(f1score3))
print('for two hidden layer, adam optimizer, 25 neurons and 1000 epoch','Accuracy:'+str(accuracy4),'precision:'+str(precision4),'recall'+str(recall4),'f1score:'+str(f1score4))

    
        