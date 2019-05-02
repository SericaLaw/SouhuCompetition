# SouhuCompetition

## 项目结构

`./src/main.py` 主程序入口

`./src/extract_entity.py` 实体抽取，基于TF-IDF

`./src/doc_encoder.py` 文章编码器，基于BERT+BiLSTM

`./src/emotion_classifier.py` 情感分类器

1. 计算各个实体向量和文章向量的余弦相似度，选取相似度最高的前三个实体（可以设置一定的阈值，只有高于阈值才被选取）
2. 将选取的各个实体向量分别和文章向量拼接，作为softmax三分类器的输入，输出情感

`./src/utils.py` 工具集，如数据预处理