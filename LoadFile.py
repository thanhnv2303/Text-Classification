import json
import os
import _pickle as cPickle
from random import randint

from gensim import corpora

import settings


class FileReader(object):
    def __init__(self, filePath, encoder=None):
        self.filePath = filePath
        self.encoder = encoder if encoder != None else 'utf-8'

    def read(self):
        with open(self.filePath) as f:
            s = f.read()
        return s

    def content(self):
        s = self.read()
        return s.encode().decode(self.encoder)

    def read_json(self):
        with open(self.filePath) as f:
            s = json.load(f)
        return s

    def read_stopwords(self):
        with open(self.filePath, 'r') as f:
            stopwords = set([w.strip().replace(' ', '_') for w in f.readlines()])
        return stopwords

    def load_dictionary(self):
        return corpora.Dictionary.load_from_text(self.filePath)


class DataLoader(object):
    def __init__(self, dataPath):
        self.dataPath = dataPath

    def __get_files(self):
        #lấy ra tất cả thư mục con trong thư mục chứa dữ liệu
        folders = [self.dataPath + folder + '/' for folder in os.listdir(self.dataPath)]
        # Lấy tên thư mục con làm nhãn cho lớp
        class_titles = os.listdir(self.dataPath)
        files = {}
        for folder, title in zip(folders, class_titles):
            # lựu lại danh sách các file trong từng thư mục con
            files[title] = [folder + f for f in os.listdir(folder)]
        self.files = files

    def get_json(self,max_file=10000):
        self.__get_files()
        data = []
        for topic in self.files:
            # rand=50
        #    rand = randint(200, 300)
            i = 0
            for file in self.files[topic]:
                # Lấy ra nội dung trong từng file của từng chủ đề
                content = FileReader(filePath=file).content()
                data.append({
                    'category': topic,
                    'content': content
                })
                if i == max_file:
                    break
                else:
                   i += 1
          #  print("Topic {} co {} văn bản".format(topic,i))
        return data
class FileStore(object):
    def __init__(self, filePath, data = None):
        self.filePath = filePath
        self.data = data

    def store_json(self):
        with open(self.filePath, 'w') as outfile:
            json.dump(self.data, outfile)

    def store_dictionary(self, dict_words):
        dictionary = corpora.Dictionary(dict_words)
        dictionary.filter_extremes(no_below=10, no_above=0.3)
        dictionary.save_as_text(self.filePath)

    def save_pickle(self, obj):
        outfile = open(self.filePath, 'wb')

        #fixing
        fastPickler = cPickle.Pickler(outfile, 0)
        fastPickler.fast = 1
        fastPickler.dump(obj)
        outfile.close()

if __name__ == '__main__':
    json_train = DataLoader(dataPath=settings.DATA_TRAIN_PATH).get_json(2999)
    FileStore(filePath=settings.DATA_TRAIN_JSON, data=json_train).store_json()

    json_test = DataLoader(dataPath=settings.DATA_TEST_PATH).get_json()
    FileStore(filePath=settings.DATA_TEST_JSON, data=json_test).store_json()