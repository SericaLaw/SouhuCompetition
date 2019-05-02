import jieba
import jieba.analyse
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)


class EntityExtractor:
    def __init__(self, k):
        self.tokenizer = BertTokenizer.from_pretrained('./assets/chinese_L-12_H-768_A-12')
        self.model = BertModel.from_pretrained('./assets/bert-base-chinese')
        self.model.eval()
        self.k = k

    def extract_entity(self, content):
        '''
        使用一定的策略抽取content中的关键词，目前使用jieba中的TF-IDF方法
        :param content:
        :return:
        '''
        entity_list = jieba.analyse.extract_tags(content, topK=2, withWeight=False)
        return entity_list

    def get_bert_encoding(self, entity):
        tokenized_text = self.tokenizer.tokenize(entity)
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, output_all_encoded_layers=False)

        encoding = encoded_layers.numpy()[0]
        n = len(encoding)
        encoding = encoding.sum(axis=0) / n
        return encoding

    def get_entity_encodings(self, content):
        entity_list = self.extract_entity(content)
        encodings = {}
        for entity in entity_list:
            encodings[entity] = self.get_bert_encoding(entity)
        return encodings


if __name__ == '__main__':
    from src.utils import load_data
    data = load_data('./data/coreEntityEmotion_example.txt')
    print(data[0]['content'])
    tokenizer = BertTokenizer.from_pretrained('./assets/chinese_L-12_H-768_A-12')
    text = data[0]['content']

    extractor = EntityExtractor(2)
    print(len(extractor.get_entity_encodings(text)['表带']))

    # a = np.array([[1,2,3],[4,5,6]])
    # print(a.sum(axis=0))

