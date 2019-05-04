import jieba
import jieba.analyse
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from src.utils import load_data
import json
import logging
logging.basicConfig(level=logging.INFO)


class EntityExtractor:
    def __init__(self, k=15):
        '''
        extract top k entities from content
        :param k: top k
        '''
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
        entity_list = jieba.analyse.extract_tags(content, topK=self.k, withWeight=False)
        return entity_list

    def get_bert_encoding(self, content):
        '''
        使用BERT对content进行character-level encoding
        :param content: a string of length n
        :return: encodings of shape (n, 768)
        '''
        tokenized_text = self.tokenizer.tokenize(content)
        # Convert token to vocabulary indices
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor, output_all_encoded_layers=False)

        encoding = encoded_layers.numpy()[0]
        return encoding


    def get_entity_encoding(self, entity):
        '''
        对entity进行编码，返回的结果为其character-level encoding的平均
        :param entity: the string of entity
        :return: an entity encoding of shape (1, 768)
        '''
        encoding = self.get_bert_encoding(entity)
        n = len(encoding)
        encoding = encoding.sum(axis=0) / n
        return encoding.reshape(1, -1)

    def get_entity_encodings(self, content):
        '''
        从content中提取k个实体，对其进行编码
        :param content: a passage
        :return: a dict like { entity1: encoding1(1, 768), entity2: encoding2(1, 768), ... }
        '''
        entity_list = self.extract_entity(content)
        encodings = []
        for entity in entity_list:
            item = dict()
            item['entity'] = entity
            item['encoding'] = self.get_entity_encoding(entity)
            encodings.append(item)
        return encodings

    def get_passages(self, items):
        passages = []
        for item in items:
            passage = dict()
            passage['title'] = self.get_bert_encoding(item['title'])
            tokenized_content = self.tokenizer.tokenize(item['content'])
            content_encoding = np.zeros((len(tokenized_content), 768))
            for idx, token in enumerate(tokenized_content):
                content_encoding[idx, :] = self.get_bert_encoding(token[0])
            content_encoding = np.array(content_encoding)
            print(content_encoding.shape)
            passage['content'] = content_encoding
            entity_encoding_list = []
            for label in item['coreEntityEmotions']:
                entity = label['entity']
                entity_encoding_list.append(
                    {
                        "entity": entity,
                        "encoding": self.get_entity_encoding(entity)
                    }
                )
            passage['entity'] = entity_encoding_list
            passage['candidate'] = self.get_entity_encodings(item['content'])
            passages.append(passage)

        return passages


def make_data_set(size=500):
    data = load_data('./data/coreEntityEmotion_train.txt')
    data = [data[i:i + size] for i in range(0, len(data), size)]
    print(len(data))
    extractor = EntityExtractor(k=5)
    for idx, items in enumerate(data):
        passages = extractor.get_passages(items)
        print(passages)
        json_passages = json.dumps(passages)
        with open('./dataset/data.txt', 'w') as f:
            f.write(json_passages)
        break


if __name__ == '__main__':

