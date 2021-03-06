import glob
import json
import numpy
import sys

fprint = open("results.out", 'w')
sys.stdout = fprint

from sentiment_classifier import SentimentClassifier
from nltk.tokenize import sent_tokenize

def get_domain(filename):
    if 'gutenberg' in filename:
        return '[LITERATURE]'
    if 'news' in filename:
        return '[NEWS]'
    if 'sports' in filename:
        return '[SPORTS]'
    else:
        return '[UNKNOWN]'

def run_test(folder, results_file_suffix):
    files = glob.glob(folder)
    pos = 0
    neg = 0
    classifier = SentimentClassifier('data/word_sentiment_data/V2_Dataset_BRM-emot-submit.csv')

    mean_valences = []
    mean_arousals = []
    sentences = []
    domains_per_doc = []

    mean_valences_per_doc = []
    mean_arousals_per_doc = []
    sentences_per_doc = []
    domains = []
    words_per_result = []

    for fle in files:
        # open the file and then call .read() to get the text
        with open(fle, "r", encoding='utf-8', errors='ignore') as f:
            domain = get_domain(f.name)
            data = f.read()
            data = data.replace('\n', ' ')
            sent_tokenize_list = sent_tokenize(data)
            result = classifier.classify_document(sent_tokenize_list)

            if result['classification'] == 'pos':
                pos += 1
            else:
                neg += 1

            mean_valences_per_doc.append(result['mean_valences'])
            mean_arousals_per_doc.append(result['mean_arousals'])
            sentences_per_doc.append([result['sentences']])
            domains_per_doc.append(domain)

            for sentence_words in result['words']:
                words_per_result.append(sentence_words)
            for sentence in result['sentences']:
                sentences.append(sentence)
                domains.append(domain)
            for mv in result['mean_valences']:
                mean_valences.append(mv)
            for ma in result['mean_arousals']:
                mean_arousals.append(ma)

    #matlab.scatter(mean_valences, mean_arousals)
    with open('out/mean_valence_and_arousals_' + results_file_suffix + ".js", 'w+', encoding='utf-8', errors='ignore') as f:
        valence_and_arousals = []
        for i in range(0, len(mean_valences)):
            m_v = mean_valences[i]
            m_a = mean_arousals[i]
            sentence = sentences[i]
            domain = domains[i]
            data_point = {
                'x': m_v,
                'y': m_a,
                'sentence': sentence,
                'domain': domain,
            }
            valence_and_arousals.append(data_point)
        f.write("var v_and_a_" + results_file_suffix + " = " + json.dumps(valence_and_arousals) + ";")


    with open('out/mean_valence_and_arousals_' + results_file_suffix + "_per_word.js", 'w+', encoding='utf-8', errors='ignore') as f:
        valence_and_arousals = []
        for i in range(0, len(mean_valences)):
            words = words_per_result[i]
            for w in words:
                data_point = {
                    'x': w['valence'],
                    'y': w['arousal'],
                    'word': w['word']
                }
                valence_and_arousals.append(data_point)
        f.write("var v_and_a_" + results_file_suffix + "_per_word = " + json.dumps(valence_and_arousals) + ";")

    with open('out/mean_valence_and_arousals_per_doc_' + results_file_suffix + '.js', 'w+', encoding='utf-8', errors='ignore') as f:
        valence_and_arousals = []
        for i in range(0, len(mean_valences_per_doc)):
            m_v = numpy.mean(mean_valences_per_doc[i])
            m_a = numpy.mean(mean_arousals_per_doc[i])
            sentence = sentences[i]
            domain = domains_per_doc[i]
            data_point = {
                'x': m_v,
                'y': m_a,
                'domain': domain
            }
            valence_and_arousals.append(data_point)
        f.write("var v_and_a_per_doc_" + results_file_suffix+ " = " + json.dumps(valence_and_arousals) + ";")

    return [pos, neg]


def report_test(pos_folder, neg_folder):
    pos_test = run_test(pos_folder, 'pos')
    neg_test = run_test(neg_folder, 'neg')

    print('POS TEST: ' + str(pos_test[0] + pos_test[1]) + ' documents')
    print('POS: ' + str(pos_test[0]) + ', NEG: ' + str(pos_test[1]))

    pos_pos = pos_test[0]
    pos_neg = pos_test[1]
    neg_neg = neg_test[1]
    neg_pos = neg_test[0]

    accuracy = (pos_pos + neg_neg) / (pos_pos + pos_neg + neg_neg + neg_pos)

    print('Precision, Recall, and F1 for Positive Sentiments:')
    precision = pos_pos / (pos_pos + neg_pos)
    recall = pos_pos / (pos_pos + pos_neg)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

    print()

    print('NEG TEST:' + str(neg_test[0] + neg_test[1]) + ' documents')
    print('POS: ' + str(neg_test[0]) + ', NEG: ' + str(neg_test[1]))

    print('Precision, Recall and F1 for Negative Sentiments:')
    precision = neg_neg / (neg_neg + pos_neg)
    recall = neg_neg / (neg_neg + neg_pos)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

    print()

    print('Total Accuracy: ' + str(accuracy))

report_test('../testing_corpus_hand_tagged/positive/*.txt', '../testing_corpus_hand_tagged/negative/*.txt')

fprint.close()