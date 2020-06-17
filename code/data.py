import torch
from torch.utils.data import DataLoader, Dataset

import numpy as np
import re
import os
import json

from config import Config
from preprocess import infinite_iter

"""# 資料結構

## 定義資料的轉換
- 將不同長度的答案拓展到相同長度，以便訓練模型
"""

class LabelTransform(object):
  def __init__(self, size, pad):
    self.size = size
    self.pad = pad

  def __call__(self, label):
    label = np.pad(label, (0, (self.size - label.shape[0])), mode='constant', constant_values=self.pad)
    return label


"""## 定義 Dataset
"""


class EN2CNDataset(Dataset):

  def __init__(self, root, max_output_len, set_name):
    self.root = root

    self.word2int_cn, self.int2word_cn = self.get_dictionary('cn')
    self.word2int_en, self.int2word_en = self.get_dictionary('en')

    # 載入資料
    self.data = []
    with open(os.path.join(self.root, f'{set_name}.txt'), "r", encoding="utf-8") as f:
      for line in f:
        self.data.append(line)
    print (f'{set_name} dataset size: {len(self.data)}')

    self.cn_vocab_size = len(self.word2int_cn)
    self.en_vocab_size = len(self.word2int_en)
    self.transform = LabelTransform(max_output_len, self.word2int_en['<PAD>'])

  def get_dictionary(self, language):
    # 載入字典
    with open(os.path.join(self.root, f'word2int_{language}.json'), "r", encoding="utf-8") as f:
      word2int = json.load(f, encoding="utf-8")
    with open(os.path.join(self.root, f'int2word_{language}.json'), "r", encoding="utf-8") as f:
      int2word = json.load(f, encoding="utf-8")
    return word2int, int2word

  def __len__(self):
    return len(self.data)

  def __getitem__(self, Index):
    # 先將中英文分開
    sentences = self.data[Index]
    sentences = re.split('[\t\n]', sentences)
    sentences = list(filter(None, sentences))
    #print (sentences)
    assert len(sentences) == 2

    # 預備特殊字元
    BOS = self.word2int_en['<BOS>']
    EOS = self.word2int_en['<EOS>']
    UNK = self.word2int_en['<UNK>']

    # 在開頭添加 <BOS>，在結尾添加 <EOS> ，不在字典的 subword (詞) 用 <UNK> 取代
    en, cn = [BOS], [BOS]
    # 將句子拆解為 subword 並轉為整數
    sentence = re.split(' ', sentences[0])
    sentence = list(filter(None, sentence))
    #print (f'en: {sentence}')
    for word in sentence:
      en.append(self.word2int_en.get(word, UNK))
    en.append(EOS)

    # 將句子拆解為單詞並轉為整數
    # e.g. < BOS >, we, are, friends, < EOS > --> 1, 28, 29, 205, 2
    sentence = re.split(' ', sentences[1])
    sentence = list(filter(None, sentence))
    #print (f'cn: {sentence}')
    for word in sentence:
      cn.append(self.word2int_cn.get(word, UNK))
    cn.append(EOS)

    en, cn = np.asarray(en), np.asarray(cn)

    # 用 <PAD> 將句子補到相同長度
    en, cn = self.transform(en), self.transform(cn)
    en, cn = torch.LongTensor(en), torch.LongTensor(cn)

    return en, cn


if __name__ == '__main__':
    config = Config()

    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    print("train data: " , next(train_iter))
    # 準備檢驗資料
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = DataLoader(val_dataset, batch_size=1)