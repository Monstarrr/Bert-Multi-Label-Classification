import random
import pandas as pd
from tqdm import tqdm
from ..common.tools import save_pickle
from ..common.tools import logger
from ..callback.progressbar import ProgressBar

class TaskData(object):
    def __init__(self):
        pass

    def train_val_split(self,X, y,valid_size,stratify=False,shuffle=True,save = True,
                        seed = None,data_name = None,data_dir = None):
        pbar = ProgressBar(n_total=len(X),desc='bucket')
        logger.info('split raw data into train and valid')
        if stratify:
            num_classes = len(list(set(y)))
            train, valid = [], []
            bucket = [[] for _ in range(num_classes)]
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                bucket[int(data_y)].append((data_x, data_y))
                pbar(step=step)
            del X, y
            for bt in tqdm(bucket, desc='split'):
                N = len(bt)
                if N == 0:
                    continue
                test_size = int(N * valid_size)
                if shuffle:
                    random.seed(seed)
                    random.shuffle(bt)
                valid.extend(bt[:test_size])
                train.extend(bt[test_size:])
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        else:
            data = []
            for step,(data_x, data_y) in enumerate(zip(X, y)):
                data.append((data_x, data_y))
                pbar(step=step)
            del X, y
            N = len(data)
            test_size = int(N * valid_size)
            if shuffle:
                random.seed(seed)
                random.shuffle(data)
            valid = data[:test_size]
            train = data[test_size:]
            # 混洗train数据集
            if shuffle:
                random.seed(seed)
                random.shuffle(train)
        if save:
            train_path = data_dir / f"{data_name}.train.pkl"
            valid_path = data_dir / f"{data_name}.valid.pkl"
            save_pickle(data=train,file_path=train_path)
            save_pickle(data = valid,file_path=valid_path)
        return train, valid

    def read_data(self,raw_data_path,preprocessor = None,is_train=True):
        '''
        :param raw_data_path:
        :param skip_header:
        :param preprocessor:
        :return:
        '''
        print("read data ...")
        with open('pybert/dataset/processed/内容类型_标签.txt', 'r', encoding="utf-8") as fp:
            labels = fp.read().strip().split("\n")
        self.tag2id = {}
        self.id2tag = {}
        for label_idx, label in enumerate(labels):
            self.tag2id[label] = label_idx
            self.id2tag[label_idx] = label
        targets, sentences = [], []
        data = open(raw_data_path, "r")
        for row in data.readlines():
            # if is_train:
            row = row.strip().split("\t")
            target = [0] * len(self.tag2id)
            labels = eval(row[0])
            if len(labels) == 0:
                continue
            for label in labels:
                label_id = self.tag2id[label]
                target[label_id] = 1
            # else:
            #     target = [-1,-1,-1,-1,-1,-1]
            sentence = str(row[1].strip())
            if preprocessor:
                sentence = preprocessor(sentence)
            if sentence:
                targets.append(target)
                sentences.append(sentence)
        return targets,sentences
