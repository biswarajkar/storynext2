import glob
import pickle

import nltk
from math import log
from collections import Counter


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


# get data from pkl
def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


def cal_uni_count(url, uni_output):
    uni_count = {}
    train_data = []
    word_count = 0

    folder = url
    files = glob.glob(folder)
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    unk_words = set()
    for fle in files:
        # open the file and then call .read() to get the text
        with open(fle, "r", encoding='utf-8', errors='ignore') as f:
            data = f.read()
            rows = tokenizer.tokenize(data)
            for row in rows:
                if row:
                    row = row[:-1].lower().split(' ')
                    # row = ['<s>'] + row + ['</s>']
                    train_data.append(row)
                    for word in row:
                        uni_count[word] = uni_count[word] + 1 if uni_count.get(word) else 1
                        word_count += 1
                        # sentences = data.replace('\n','').split('.')
                        # print(sentences)
    unk_times = 0
    for k in uni_count.keys():
        if uni_count[k] < 5:
            unk_words.add(k)
            unk_times += uni_count[k]
    for word in unk_words:
        del uni_count[word]
    uni_count['<UNK>'] = unk_times
    save_obj(uni_count, uni_output)
    return word_count


def generate_conditional_prob(class_name, count, V):
    conditional_prob = {}
    dict = load_obj(class_name + '_uni_count')
    for k, v in dict.items():
        conditional_prob[(k, class_name)] = (v + 1) / (count + V)
    conditional_prob[('<UNK>', class_name)] = 1 / (count + V)
    save_obj(conditional_prob, class_name + '_conditional_prob')


def naive_classification(article):
    class_point = {}
    class_conditional_prob = {}

    docs = load_obj('doc_count')
    for k in docs:
        class_point[k] = 0
        class_conditional_prob[k] = load_obj(k + '_conditional_prob')
        # print(row)
    class_P = docs.copy()
    # print(class_P)
    articleList = article.lower().replace('\n', ' ').split(' ')
    # print(articleList)
    for key, P in class_P.items():
        # print(key + ':')
        # pair = {}
        prob = class_conditional_prob[key]
        for word in articleList:
            # print(word)
            if (word, key) not in prob:
                P += log(prob[('<UNK>', key)])
            else:
                P += log(prob[(word, key)])
        # print(pair)
        class_P[key] = P
    # print(class_point)
    return key_for_max_value(class_P)


def key_for_max_value(dict):
    v = list(dict.values())
    k = list(dict.keys())
    return k[v.index(max(v))]


def training(classes, docs):
    totalNum = 0
    totalDict = Counter({})
    for k, v in classes.items():
        url = '../Dataset/train/' + k + '/*.txt'
        uni_count = k + '_uni_count'
        classes[k] = cal_uni_count(url, uni_count)
    for k, v in classes.items():
        totalDict += Counter(load_obj(k + '_uni_count'))
        totalNum += v
    save_obj(totalDict, 'total_word_count')
    for k, v in classes.items():
        generate_conditional_prob(k, v, len(totalDict))
    s = sum(docs.values())
    for k, v in docs.items():
        docs[k] = v / s
    save_obj(docs, 'doc_count')
