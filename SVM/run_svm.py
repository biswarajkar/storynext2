from sentiment_classification import run
import sys

print("Running SVM and Random Forest using 25K reviews IMDB corpus and redirecting output to 'results.out'")

f = open("results.out", 'w')
sys.stdout = f

dir = 'Dataset'
run(dir)

f.close()