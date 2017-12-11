#!/usr/bin/python

import cgi, cgitb
import traceback
from collections import Counter

cgitb.enable()
import json
import csv
import numpy

from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

import math

  # for troubleshooting

########


def weighted_mean(means_and_stds_and_freqs):
    total_weight = 0
    means_and_pfds_and_freqs = []
    for mean_and_std_and_freq in means_and_stds_and_freqs:
        mean = mean_and_std_and_freq[0]
        std = mean_and_std_and_freq[1]
        freq = mean_and_std_and_freq[2]
        total_weight += freq
        means_and_pfds_and_freqs.append([mean, probabalistic_density_function_of_mean(std), freq])

    # Weighted mean = Sum(pfd * mean) / Sum(pfd)
    numerator = 0
    denominator = 0
    running_total = 0
    for mean_and_pfd_and_freq in means_and_pfds_and_freqs:
        mean = mean_and_pfd_and_freq[0]
        pfd = mean_and_pfd_and_freq[1]
        freq = mean_and_pfd_and_freq[2]
        running_total += (freq / total_weight) * mean
        numerator += mean * pfd
        denominator += pfd

    return 0 if numerator == 0 else (numerator / denominator)


def probabalistic_density_function_of_mean(std):
    return 1 / math.sqrt(2 * math.pi * std * std)

# Valence is between 1 and 9. To reverse, we flip over 5:
# e.g. 1 becomes 9, and 7.5 becomes 3.5
def reverse_valence(val):
    return 5 - (val - 5)

class SentimentClassifier:

    def __init__(self, file):
        self.data = self.load_csv(file)

    def load_csv(self, file):
        with open(file) as csvfile:
            reader = csv.DictReader(csvfile)
            words = {}
            for row in reader:
                row_data = {}
                word = row['Word']
                row_data['v_mean_sum'] = row['V.Mean.Sum']
                row_data['v_sd_sum'] = row['V.SD.Sum']
                row_data['v_freq_sum'] = row['V.Rat.Sum']

                row_data['a_mean_sum'] = row['A.Mean.Sum']
                row_data['a_sd_sum'] = row['A.SD.Sum']
                row_data['a_freq_sum'] = row['A.Rat.Sum']
                words[word] = row_data

            return words


    # Input: List of sentences (text)
    def classify_document(self, sentences):
        all_sentence_mean_valences = []
        all_sentence_mean_arousals = []
        all_sentences = []
        all_words_per_sentence = []

        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            v_means_and_stds_and_freqs = []
            a_means_and_stds_and_freqs = []
            words_v_and_a = []
            for idx, word in enumerate(words):
                data = self.data
                if word in data:
                    word_data = data[word]
                    v_mean_sum = float(word_data['v_mean_sum'])

                    # if previous word is a 'not' token, then reverse the valence
                    if (idx > 0 and self.is_not_token(words[idx - 1])):
                        v_mean_sum = reverse_valence(v_mean_sum)

                    v_sd_sum = float(word_data['v_sd_sum'])
                    v_freq_sum = int(word_data['v_freq_sum'])
                    v_means_and_stds_and_freqs.append([v_mean_sum, v_sd_sum, v_freq_sum])

                    a_mean_sum = float(word_data['a_mean_sum'])
                    a_sd_sum = float(word_data['a_sd_sum'])
                    a_freq_sum = int(word_data['a_freq_sum'])
                    a_means_and_stds_and_freqs.append([a_mean_sum, a_sd_sum, a_freq_sum])

                    words_v_and_a.append({'word': word, 'arousal': a_mean_sum, 'valence': v_mean_sum})

            if v_means_and_stds_and_freqs and a_means_and_stds_and_freqs:
                all_words_per_sentence.append(words_v_and_a)
                all_sentences.append(sentence)
                v_weighted_mean = weighted_mean(v_means_and_stds_and_freqs)
                a_weighted_mean = weighted_mean(a_means_and_stds_and_freqs)
                all_sentence_mean_valences.append(v_weighted_mean)
                all_sentence_mean_arousals.append(a_weighted_mean)

        return {
            'classification': self.classify(all_sentence_mean_valences, all_sentence_mean_arousals),
            'mean_valences': all_sentence_mean_valences,
            'mean_arousals': all_sentence_mean_arousals,
            'words': all_words_per_sentence,
            'sentences': all_sentences
        }

    def is_not_token(self, token):
        return token in ["n't", "not", "no", "never", "neither"]

    # To classify, we start with a naive approach: Use mean of mean valences
    # mean_valences/mean_arousals are lists of means, with each mean coming from 1 sentence.
    def classify(self, mean_valences, mean_arousals):
        mean_valence = numpy.mean(mean_valences)
        return 'pos' if mean_valence >= 5 else 'neg'



    def for_raj(self, document_str):
        result = {
            'positive': [],
            'negative': []
        }
        words = word_tokenize(document_str.lower())
        for idx, word in enumerate(words):
            if word in result['positive'] or word in result['negative']:
                continue

            data = self.data
            if word in data:
                word_data = data[word]
                v_mean_sum = float(word_data['v_mean_sum'])
                a_mean_sum = float(word_data['a_mean_sum'])

                # if previous word is a 'not' token, then reverse the valence
                if (idx > 0 and self.is_not_token(words[idx - 1])):
                    v_mean_sum = reverse_valence(v_mean_sum)

                if self.classify([v_mean_sum], [a_mean_sum]) == 'pos':
                    result['positive'].append(word)
                else:
                    result['negative'].append(word)

        return result




########
# andrew's test
try:
    classifier = SentimentClassifier('../data/V2_Dataset_BRM-emot-submit.csv')
    raw_text = cgi.FieldStorage()["content"].value
    sent_tokenize_list = sent_tokenize(raw_text)
    result = classifier.classify_document(sent_tokenize_list)

    mean_valences = []
    mean_arousals = []
    sentences = []

    mean_valences_per_doc = []
    mean_arousals_per_doc = []
    sentences_per_doc = []
    words_per_result = []

    mean_valences_per_doc.append(result['mean_valences'])
    mean_arousals_per_doc.append(result['mean_arousals'])
    sentences_per_doc.append([result['sentences']])

    for sentence_words in result['words']:
        words_per_result.append(sentence_words)
    for sentence in result['sentences']:
        sentences.append(sentence)
    for mv in result['mean_valences']:
        mean_valences.append(mv)
    for ma in result['mean_arousals']:
        mean_arousals.append(ma)

    to_return = {
        'sentiment_over_time': {
            'labels': [],
            'data': []
        },
        'sentiment_per_word': {
            'labels': [],
            'data': [],
            'texts': []
        },
        'sentiment_per_sentence': {
            'labels': [],
            'data': [],
            'texts': []
        }
    }

    for i in range(0, len(mean_valences)):
        m_v = mean_valences[i]
        m_a = mean_arousals[i]
        sentence = sentences[i]
        words = words_per_result[i]

        to_return['sentiment_per_sentence']['labels'].append(m_v)
        to_return['sentiment_per_sentence']['data'].append(m_a)
        to_return['sentiment_per_sentence']['texts'].append(sentence)

        to_return['sentiment_over_time']['labels'].append(i)
        to_return['sentiment_over_time']['data'].append(m_v)

        # sentiment_per_word
        for w in words:
            to_return['sentiment_per_word']['labels'].append(w['valence'])
            to_return['sentiment_per_word']['data'].append(w['arousal'])
            to_return['sentiment_per_word']['texts'].append(w['word'])

except:
    traceback.print_exc()



#the cgi library gets vars from html
data = cgi.FieldStorage()
# print "Content-Type: text/html\n\n"
print "Content-Type: application/json\n\n"
# print "The foo data is: " + data["content"].value
print json.dumps(to_return)
#this is the actual output
# print "Content-Type: text/html\n"

# print "<br />"
# print "The bar data is: " + data["bar"].value
# print "<br />"