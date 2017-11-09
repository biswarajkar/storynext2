from rnn_preprocess_data import maxReviewWordLength, np
from random import randint

training_ids = np.load('vectorizedTrainingMatrix.npy')
test_ids = np.load('vectorizedTestMatrix.npy')

# Use 23000 Reviews (Review# 1 to 11500 and 13500 to 25000) as Training
def getTrainBatch(batchSize):
    labels = []
    arr = np.zeros([batchSize, maxReviewWordLength])
    for i in range(batchSize):
        if (i % 2 == 0):
            # Use 11.5k reviews (Review# 1 to 11500) as Positive Training
            num = randint(1, 11499)
            labels.append([1, 0])
        else:
            # Use 11.5k reviews (Review# 13500 to 25000) as Negative Training
            num = randint(13499, 24999)
            labels.append([0, 1])
        arr[i] = training_ids[num - 1:num]
    return arr, labels


# Use 2000 Movie Reviews (Review# 11500 to 13500) as Test
def getTestBatch(batchSize):
    labels = []
    arr = np.zeros([batchSize, maxReviewWordLength])
    for i in range(batchSize):
        num = randint(11499, 13499)
        if (num <= 12499):
            # Use 1000 reviews (Review# 11500 to 12499) as Positive Test
            labels.append([1, 0])
        else:
            # Use 1000 reviews (Review# 12500 to 13500) as Negative Test
            labels.append([0, 1])
        arr[i] = training_ids[num - 1:num]
    return arr, labels

# Use hand-tagged data as Test
def getCustomTestBatch(batchSize):
    test_labels = np.load('trueTestLabels.npy')

    labels = []
    arr = np.zeros([batchSize, maxReviewWordLength])
    for i in range(batchSize):
        num = randint(0, len(test_labels))
        if (test_labels[num] == 1):
            labels.append([1, 0])
        else:
            labels.append([0, 1])
        arr[i] = test_ids[num - 1:num]
    return arr, labels
