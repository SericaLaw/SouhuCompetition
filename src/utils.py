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
    similarity_list = []
    for entity in entity_candidate:
        simi = similarity(passage_encode, entity)
        similarity_list.append((simi,entity))
    sorted(similarity_list, key= lambda x: x[0])
    return similarity_list[0:3]


def similarity(passage_encode, entity):
    simi = nn.functional.cosine_similarity(passage_encode, entity)
    return simi



