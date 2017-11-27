from ValenceArousal.sentiment_classifier import SentimentClassifier


classifier = SentimentClassifier('../data/word_sentiment_data/V2_Dataset_BRM-emot-submit.csv')

# Returns a map:
# {
#   'positive': [],
#   'negative': []
# }
#
# where each list is a list of words.
output_file = 'example_filename'
result = classifier.for_raj('lalala this is a string bad bad good not ok yay happy sad')

with open('out/' + output_file + '_pos.txt', 'w+', encoding='utf-8', errors='ignore') as f:
    positive_words = result['positive']
    f.write('\n'.join(positive_words))
    f.write('\n')

with open('out/' + output_file + '_neg.txt', 'w+', encoding='utf-8', errors='ignore') as f:
    negative_words = result['negative']
    f.write('\n'.join(negative_words))
    f.write('\n')