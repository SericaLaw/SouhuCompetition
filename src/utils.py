import json
import codecs
import torch
import torch.nn as nn


def load_data(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')
    data = []
    for line in f.readlines():
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        news = json.loads(line.strip())
        data.append(news)
    return data


def top3entity(passage_encode, entity_candidate):
    '''

    :param passage_encode: a tensor of shape (hidden_size)
    :param entity_candidate: a list of  dict in type {"entity": "a", "encoding": torch.tensor}
    :return:
    '''
    similarity_list = []
    for entity in entity_candidate:
        simi = similarity(passage_encode, entity["encoding"])
        entity["similarity"] = simi
        similarity_list.append(entity)
    sorted(similarity_list, key= lambda x: x[0])
    return similarity_list[0:3]


def similarity(passage_encode, entity):
    simi = torch.dist(passage_encode,entity,p=2)
    return simi



def F1Score(truth_prediction_list):
    '''
    [{passage: [grand truth list], y_hat: [prediction list]}, ... ]
    :return: F1 Score
    '''

    true_possitive = false_positive = false_negative = 0
    for d in truth_prediction_list:
        truth = d['passage']
        y_hat = d['y_hat']
        for t in truth:
            if t in y_hat:
                true_possitive += 1
            else:
                false_negative += 1
        for h in y_hat:
            if h not in truth:
                false_positive += 1
    precision = true_possitive / (true_possitive + false_positive)
    recall = true_possitive / (true_possitive + false_negative)
    return 2 * precision * recall / (precision + recall)


if __name__ == '__main__':
    test = [{'passage': ['a', 'b', 'c'], 'y_hat': ['a', 'b', 'c']},
            {'passage': ['d', 'e', 'f'], 'y_hat': ['d', 'e']}]
    print(F1Score(test))