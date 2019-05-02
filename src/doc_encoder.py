import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class PassageEncoder(nn.Module):
    def __init__(self, input_size, output_size):
        '''
        initial encoder
        :param input_size: input size of lstm
        :param output_size:  output size of lstm
        '''
        self.bilstm = nn.LSTM(input_size=input_size,hidden_size=output_size,
                              dropout=0.33,bidirectional=True)

    def forward(self, feature, input_length, hidden=None):
        '''
        :param feature: torch tensor of shape(batch_size, )
        :param input_length: length of input, torch tensor of dtype int
        :param hidden: initial hidden state, torch tensor shape(output_size)
        :return: the output of lstm of shape(batch_size, length, output_size)
        '''
        packed = nn.utils.rnn.pack_padded_sequence(feature, batch_first=True)
        lstm_out,hidden = self.bilstm(packed, hidden)
        out = nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        return out


class PassageDataset(Dataset):
    def __init__(self, passages):
        '''
        :param passages: a list contain dict with keys title, content, label and entity candidate
                        title and content shape in (d, length)
        '''
        self.all_passage = []
        for passage in passages:
            title = passage["title"]
            content = passage["content"]
            label = passage["label"]
            entity_candidata = passage["candidate"]
            passage_vector = self.aggregate_title_content(title,content)
            length = passage_vector.shape[1]
            self.all_passage.append({"passage":passage_vector,
                                     "label":label,
                                     "length":length,
                                     "entity_candidate":entity_candidata})

    def __len__(self):
        return len(self.all_passage)

    def __getitem__(self, item):
        passage = self.all_passage[item]
        return passage

    def aggregate_title_content(self, title, content):
        '''
        aggregate title and content to form passage vector
        here just cat content after title
        :param title: torch tensor of shape (d, title_len)
        :param content: torch tensor of shape (d, content_len)
        :return: aggregated passage tensor of shape(d, title_len+content_len)
        '''
        passage = torch.cat([title,content],dim=1)
        return passage


def loss(encoded_passage, label, entity_candidate):
    '''
    triplet loss with anchor as encoded_passage, positive as label, negtive as entity_candidate
    :param encoded_passage: output of lstm with shape ()
    :param label: the passage label of 3 entity
    :param entity_candidate: generated by tf-idf
    :return: loss
    '''

    return




def train(dataloader, model):
    model.train()
    NUM_EPOCH = 10
    LEARNING_RATE = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    for step in range(NUM_EPOCH):
        for idx, data in enumerate(dataloader):
            passage, length, label = data["passage"], data["length"], data["label"]
            data_sort_by_len = torch.sort(length, descending=True)
            sorted_passage = passage[data_sort_by_len[1].tolist()]
            sorted_label = label[data_sort_by_len[1].tolist()]
            encoded_passage, _ = model(sorted_passage, length)
            loss = loss(encoded_passage, )



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