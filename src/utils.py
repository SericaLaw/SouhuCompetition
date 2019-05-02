import torch
import torch.nn as nn


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