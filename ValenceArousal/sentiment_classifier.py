import csv
import numpy

from nltk.tokenize import word_tokenize
from formulas import weighted_mean
from formulas import reverse_valence
from nltk.tokenize import sent_tokenize

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



