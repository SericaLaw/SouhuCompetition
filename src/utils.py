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



