Python version: Python 3.6.3
Tensorflow version: 1.3.0
matplotlib version: 2.1.0
jupyter	version: 1.0.0

Training Size: 25000 reviews: 12500 Positive and 12500 Negative Reviews (IMDB)
Test Size: 151 Hand tagged Texts (76 Positive + 75 Negative)
Model Params: 50000 iterations, 64 LSTM units, Gradient Descent Optimizer with Learning rate of 0.1

Please run the file tf_lstm_train_test_eval.py to classify using a Recursive Neutral Network model trained on LSTMs.


Notes about commented Code:
---------------------------
rnn_vectorize_training.py:
Lines 20-68: Commented so that we can use existing Word Vectors, Uncomment to create the Word Vectors from fresh Training data.
Vectorization may take up to 1 hour.

rnn_vectorize_test.py:
Lines 80-135 : Can be commented to skip test Word Vector generation, takes minutes for the 150 hand annotated test set.

tf_lstm_train_test_eval.py:
Lines 73-123 :Commented to use existing Trained model, Uncomment to train the model fresh
WARNING: Training takes 7-8 hours for about 70000 iterations
