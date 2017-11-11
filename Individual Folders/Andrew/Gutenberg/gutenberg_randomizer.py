import glob
import random

from Daniel.NaiveBayesClassification.training import naive_classification

def run_test(folder):
    files = glob.glob(folder)
    pos = 0
    neg = 0
    for fle in files:
        # open the file and then call .read() to get the text
        with open(fle, "r", encoding='utf-8', errors='ignore') as f:
            data = f.read()
            data = data.replace('\n', ' ')
            if 'pos' == naive_classification(data):
                pos += 1
            else:
                neg += 1

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

    print('Precision, Recall, and F1 for POSITIVE test:')
    precision = pos_pos / (pos_pos + neg_pos)
    recall = pos_pos / (pos_pos + pos_neg)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

    print()

    print('Precision, Recall, and F1 for NEGATIVE test:')
    precision = neg_neg / (neg_neg + pos_neg)
    recall = neg_neg / (neg_neg + neg_pos)
    f1 = (2 * precision * recall) / (precision + recall)
    print('Precision: ' + str(precision))
    print('Recall: ' + str(recall))
    print('F1: ' + str(f1))

# single paragraphs
report_test('data/test/gutenberg_raw/pos/*.txt', 'data/test/gutenberg_raw/neg/*.txt')

# multi paragraphs
report_test('data/test/gutenberg_raw/paragraph_count/4/pos/*.txt', 'data/test/gutenberg_raw/paragraph_count/4/neg/*.txt')

# Given a text file and some int, save n paragraphs from the text file
# Paragraphs delineated via newline tokens '\n'
def save_random_paragraphs(folder, n, n_paragraphs):
    files = glob.glob(folder)
    paragraphs = []
    for fle in files:
        with open(fle, "r", encoding='utf-8', errors='ignore') as f:
            running_paragraph = ''
            for line in f:
                if line == '\n':
                    if running_paragraph != '' and running_paragraph != '\n':
                        paragraphs.append(running_paragraph)
                        running_paragraph = ''
                else:
                    running_paragraph += ' ' + line

    # now choose n paragraphs and save them
    chosen_random_numbers = []
    for x in range(n):
        random_number = random.randint(0,len(paragraphs) - n_paragraphs)
        while (random_number in chosen_random_numbers):
            random_number = random.randint(0,len(paragraphs) - n_paragraphs)
        with open('data/test/gutenberg_raw/unannotated/gutenberg_text_' + str(random_number) + '.txt', 'w+', encoding='utf-8') as f:
            # concatenate the paragraphs
            p = paragraphs[random_number:random_number + n_paragraphs]
            text = ''
            for t in p:
                text += t + '\n'
            f.write(text)
            f.close()


# UNCOMMENT TO ADD MORE TEST PARAGRAPHS
#save_random_paragraphs('data/test/gutenberg_raw/sources/*.txt', 50, 4)