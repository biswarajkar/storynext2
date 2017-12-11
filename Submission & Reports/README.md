Python version used: Python 3.6.3

The first step to run the project is to create a virtual environment in Python and install all dependent libraries. All dependent libraries are listed in requirements.txt.

One way to do it is start with Tensorflow setup:
Create a folder called “tensorflow” in your default user folder
Run *pip3 install --upgrade virtualenv*
Run *virtualenv --system-site-packages -p python3 ~/tensorflow*
Run *source ~/tensorflow/bin/activate*
Prompt changes to *(tensorflow)$*
Then use the *requirements.txt* to install all required libraries into this virtual env. 

Additional Help for specific Libraries:
Tensorflow (v1.3): https://www.tensorflow.org/install/
Virtual Environment: https://virtualenv.pypa.io/en/stable/
Scikit: http://scikit-learn.org/stable/install.html
matplotlib: https://matplotlib.org/users/installing.html
NLTK: http://www.nltk.org/install.html


The project has the following modules:
1. Naive Bayes
	Please run the file "run_nb.py" to classify using Naive Bayes classification
        *OR* follow the instructions in the README file in the 'NaiveBayes' folder
2. SVM & Random Forest
	Please run the file "run_svm.py" to classify using SVM and Random Forest
        *OR* follow the instructions in the README file in the 'SVM' folder
3. RNN LSTM
	Please run the file "tf_lstm_train_test_eval.py" to classify using RNN/LSTM pertained models
        *OR* follow the instructions in the README file in the 'RNN_LSTM' folder
4. Web Interface
	Please run the index_final.html file after placing the whole project in the deployment folder of any Python-CGI-supported application server. We used MAMP for the same.
	_*This part is still work in progress, please check the Project Wiki page at https://github.com/biswarajkar/storynext2/wiki for the latest setup instructions.*_
