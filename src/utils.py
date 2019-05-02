import json
import codecs


def load_data(filePath):
    f = codecs.open(filePath, 'r', 'utf-8')

    data = []
    for line in f.readlines():
        if line.startswith(u'\ufeff'):
            line = line.encode('utf8')[3:].decode('utf8')
        news = json.loads(line.strip())
        data.append(news)
    return data