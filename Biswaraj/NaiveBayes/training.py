import glob
import pickle
import nltk
from math import log
from collections import Counter


# Create a Unigram model
def get_unigrams_from_path(url, unigrams):
    unigram_counts = {}
    training_data = []
    tot_unigram_count = 0

    files = glob.glob(url)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    unk_words = set()
    for fle in files:
        with open(fle, "r", encoding='utf-8', errors='ignore') as f:
            data = f.read()
            rows = tokenizer.tokenize(data)
            for row in rows:
                if row:
                    row = row[:-1].lower().split(' ')
                    training_data.append(row)
                    for word in row:
                        unigram_counts[word] = unigram_counts[word] + 1 if unigram_counts.get(word) else 1
                        tot_unigram_count += 1
    unk_times = 0
    for doc in unigram_counts.keys():
        # Add UNK for < 5 occurences
        if unigram_counts[doc] < 5:
            unk_words.add(doc)
            unk_times += unigram_counts[doc]

    # Remove UNK tagged from word_list and put UNK as a word with total count
    for word in unk_words:
        del unigram_counts[word]
    unigram_counts['<UNK>'] = unk_times

    # Save Unigram data
    save_data(unigram_counts, unigrams)
    return tot_unigram_count


# Main Training function
def training(classes, doc_count_in_class):
    total_num = 0
    total_counts = Counter({})

    # Process input and compute Unigrams
    for true_classes, true_counts in classes.items():
        url = 'data/train/' + true_classes + '/*.txt'
        unigram_count = true_classes + '_unigram_count'
        classes[true_classes] = get_unigrams_from_path(url, unigram_count)
        total_counts = total_counts + Counter(get_data(true_classes + '_unigram_count'))
        total_num = total_num + true_counts

    save_data(total_counts, 'word_count')

    # Calculate MLE Probability of all Words using Laplace Smoothing
    for doc, cnt in classes.items():
        mle_laplace_prob = {}
        vocab_len = len(total_counts)

        # Get the counts for the Class passed
        dict = get_data(doc + '_unigram_count')

        for doc_num, val in dict.items():
            mle_laplace_prob[(doc_num, doc)] = (val + 1) / (cnt + vocab_len)
            mle_laplace_prob[('<UNK>', doc)] = 1 / (cnt + vocab_len)

        # Save after calculating MLE
        save_data(mle_laplace_prob, doc + '_mle_laplace_prob')

    s = sum(doc_count_in_class.values())

    for doc_num, cnt in doc_count_in_class.items():
        doc_count_in_class[doc_num] = cnt / s

    save_data(doc_count_in_class, 'total_doc_counts')


# Main Classifier Function
def classifier_nb(doc_data):
    class_score = {}
    class_conditional_prob = {}

    docs = get_data('total_doc_counts')
    for doc in docs:
        class_score[doc] = 0
        class_conditional_prob[doc] = get_data(doc + '_mle_laplace_prob')

    class_p = docs.copy()
    doc_list = doc_data.lower().replace('\n', ' ').split(' ')
    for doc, doc_prob in class_p.items():
        prob = class_conditional_prob[doc]
        for word in doc_list:
            if (word, doc) not in prob:
                doc_prob += log(prob[('<UNK>', doc)])
            else:
                doc_prob += log(prob[(word, doc)])
        class_p[doc] = doc_prob

    docs_classifed = list(class_p.values())
    labels = list(class_p.keys())
    most_linkely = labels[docs_classifed.index(max(docs_classifed))]
    return most_linkely


# Data Persistence functions
# Save Data
def save_data(data_ref, filename):
    with open('obj/' + filename + '.pkl', 'wb') as file:
        pickle.dump(data_ref, file, pickle.HIGHEST_PROTOCOL)


# Load Saved Data
def get_data(filename):
    with open('obj/' + filename + '.pkl', 'rb') as file:
        return pickle.load(file)
