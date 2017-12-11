from sentiment_classification import run
import sys

f = open("results.out", 'w')
sys.stdout = f

dir = 'Dataset'
run(dir)

f.close()