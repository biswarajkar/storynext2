Python version: Python 3.6.3
Tensorflow version: 1.3.0
tensorflow-tensorboard version:	0.1.8 (optional, for Hyper-parameter visualization)
matplotlib version: 2.1.0
NLTK version: 3.2.5
numpy version: 1.13.3

Training Size: 25000 reviews: 12500 Positive and 12500 Negative Reviews (IMDB)
Test Size: 151 Hand tagged Texts (76 Positive + 75 Negative)
Model Params: 70000 iterations, 64 LSTM units, Gradient Descent Optimizer with Learning rate of 0.1

Please run the file "tf_lstm_train_test_eval.py" to classify using a RNN model with LSTMs.

Notes about commented Code:
---------------------------
rnn_vectorize_training.py:
Lines 20-68: Commented so that we can use existing Word Vectors, Uncomment to create the Word Vectors from fresh Training data.
NOTE: Vectorization may take up to 1 hour.

rnn_vectorize_test.py:
Lines 80-135 : Can be commented to skip test Word Vector generation, takes seconds for the 150 hand annotated test set and hence is not commented.

tf_lstm_train_test_eval.py:
Lines 73-123 : Commented to use existing Trained model, Uncomment to train the model fresh on the training data.
[WARNING: Training takes 8-10 hours on a Mac/PC for about 70000 iterations]

Code Execution Flow:
--------------------

tf_lstm_train_test_eval.py [Entry Point/Controller]
|---> 1. rnn_preprocess_data.py
|-------> 2. rnn_vectorize_training.py
|----------> 3. tf_support_functions.py
|----------> 4. rnn_vectorize_test.py
