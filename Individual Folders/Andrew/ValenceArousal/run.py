import glob
import matlab
PACKAGE_PARENT = '.'
from .sentiment_classifier import SentimentClassifier
from nltk.tokenize import sent_tokenize


def run_test(folder):
    files = glob.glob(folder)
    pos = 0
    neg = 0
    classifier = SentimentClassifier('data/word_sentiment_data/V2_Dataset_BRM-emot-submit.csv')

    mean_valences = []
    mean_arousals = []
    for fle in files:
        # open the file and then call .read() to get the text
        with open(fle, "r", encoding='utf-8', errors='ignore') as f:
            data = f.read()
            data = data.replace('\n', ' ')
            sent_tokenize_list = sent_tokenize(data)
            result = classifier.classify_document(sent_tokenize_list)

            if result['classification'] == 'pos':
                pos += 1
            else:
                neg += 1

            for mv in result['mean_valences']:
                mean_valences.append(mv)
            for ma in result['mean_arousals']:
                mean_arousals.append(ma)

    matlab.scatter(mean_valences, mean_arousals)
    return [pos, neg]


def report_test(pos_folder, neg_folder):
    pos_test = run_test(pos_folder)
    neg_test = run_test(neg_folder)

    print('POS TEST: ' + str(pos_test[0] + pos_test[1]) + ' documents')
    print('POS: ' + str(pos_test[0]) + ', NEG: ' + str(pos_test[1]))

    print('NEG TEST:' + str(neg_test[0] + neg_test[1]) + ' documents')
    print('POS: ' + str(neg_test[0]) + ', NEG: ' + str(neg_test[1]))

    pos_pos = pos_test[0]
    pos_neg = pos_test[1]
    neg_neg = neg_test[1]
    neg_pos = neg_test[0]

    print('Precision, Recall, and F1 for POSITIVE test_gitignore:')
    precision = pos_pos / (pos_pos + neg_pos)
    recall = pos_pos / (pos_pos + pos_neg)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

    print()

    print('Precision, Recall, and F1 for NEGATIVE test_gitignore:')
    precision = neg_neg / (neg_neg + pos_neg)
    recall = neg_neg / (neg_neg + neg_pos)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))


report_test('data/test_gitignore/positive/*.txt', 'data/test_gitignore/negative/*.txt')