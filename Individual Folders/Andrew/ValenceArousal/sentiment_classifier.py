import csv
import numpy

from nltk.tokenize import word_tokenize
from Andrew.ValenceArousal.formulas import weighted_mean


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
                row_data['a_mean_sum'] = row['A.Mean.Sum']
                row_data['a_sd_sum'] = row['A.SD.Sum']
                words[word] = row_data

            return words


    # Input: List of sentences (text)
    def classify_document(self, sentences):
        all_sentence_mean_valences = []
        all_sentence_mean_arousals = []

        for sentence in sentences:
            words = word_tokenize(sentence.lower())
            v_means_and_stds = []
            a_means_and_stds = []
            for word in words:
                data = self.data
                if word in data:
                    word_data = data[word]
                    v_mean_sum = float(word_data['v_mean_sum'])
                    v_sd_sum = float(word_data['v_sd_sum'])
                    v_means_and_stds.append([v_mean_sum, v_sd_sum])

                    a_mean_sum = float(word_data['a_mean_sum'])
                    a_sd_sum = float(word_data['a_sd_sum'])
                    a_means_and_stds.append([a_mean_sum, a_sd_sum])

            v_weighted_mean = weighted_mean(v_means_and_stds)
            a_weighted_mean = weighted_mean(a_means_and_stds)
            all_sentence_mean_valences.append(v_weighted_mean)
            all_sentence_mean_arousals.append(a_weighted_mean)

        return {
            'classification': self.classify(all_sentence_mean_valences, all_sentence_mean_arousals),
            'mean_valences': all_sentence_mean_valences,
            'mean_arousals': all_sentence_mean_arousals
        }


    # To classify, we start with a naive approach: Use mean of mean valences
    # mean_valences/mean_arousals are lists of means, with each mean coming from 1 sentence.
    def classify(self, mean_valences, mean_arousals):
        mean_valence = numpy.mean(mean_valences)
        return 'pos' if mean_valence >= 5 else 'neg'

