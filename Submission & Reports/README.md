## Team 6 - StoryNext 2.0    
**Biswaraj Kar | Andrew Krischer | Ling Zhang**  
The project is coded with Python version 3.6.3 using PyCharm CE as the IDE

The first step to run the project is to **create a [virtual environment](https://virtualenv.pypa.io/en/stable/) in Python and install all dependent libraries.**   
All dependent libraries are listed in **requirements.txt** in the root of the project folder.

One way to do it is start with Tensorflow setup:   
- Create a folder called “tensorflow” in your default user folder    
- Run **pip3 install --upgrade virtualenv**  
- Run **virtualenv --system-site-packages -p python3 ~/tensorflow**    
- Run **source ~/tensorflow/bin/activate**    
- Prompt changes to **(tensorflow)$**   
- Then use the _**requirements.txt**_ to install all required libraries into this virtual env. by running the command **pip3 install -r requirements.txt**

Additional Help for specific Libraries:    
- Tensorflow (v1.3): https://www.tensorflow.org/install/    
- Virtual Environment: https://virtualenv.pypa.io/en/stable/    
- Scikit: http://scikit-learn.org/stable/install.html    
- matplotlib: https://matplotlib.org/users/installing.html    
- NLTK: http://www.nltk.org/install.html

The project has the following modules:    
1. Multinomial Naive Bayes    
        Please run the file "run_nb.py" to classify using Naive Bayes classification    
	**OR** follow the instructions in the README file in the 'NaiveBayes' folder    
2. SVM & Random Forest    
     Please run the file "run_svm.py" to classify using SVM and Random Forest.    
     **OR** follow the instructions in the README file in the 'SVM' folder.    
3. RNN LSTM    
     Please run the file "tf_lstm_train_test_eval.py" to classify using RNN/LSTM pertained models.    
     **OR** follow the instructions in the README file in the 'RNN_LSTM' folder.    
4. Valence & Arousal 
     Please run the file "run_valence_arousal.py" to classify using the Valence & Arousal model.    
     **OR** follow the instructions in the README file in the 'ValenceArousal' folder.    
5. Web Interface    
     Please run the index_final.html file after placing the whole project in the deployment folder of any Python-CGI-supported web-application server. We used MAMP for the same.    
     **This UI part is still work in progress, please check the Project Wiki page at https://github.com/biswarajkar/storynext2/wiki for the latest setup instructions and updates**
