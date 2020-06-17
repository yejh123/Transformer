"""# Config
- 實驗的參數設定表
"""
import os
import torch


class Config(object):
    def __init__(self):
        self.device = torch.device('cuda')
        self.DEBUG = True
        self.batch_size = 100
        self.n_layers = 4
        self.model_dim = 512
        self.num_heads = 8
        self.ffn_dim = 2048
        self.dropout = 0.2
        self.learning_rate = 0.000001
        self.max_output_len = 40  # 最後輸出句子的最大長度
        self.num_steps = 12000  # 總訓練次數
        self.store_steps = 300  # 訓練多少次後須儲存模型
        self.summary_steps = 300  # 訓練多少次後須檢驗是否有overfitting
        self.load_model = True  # 是否需載入模型
        self.store_model_path = "../ckpt_v0.3"  # 儲存模型的位置
        self.model_file = "model_3300.ckpt"
        self.load_model_path = os.path.join(self.store_model_path, self.model_file)  # 載入模型的位置 e.g. "./ckpt_v0.2/model_{step}"
        self.data_path = "../cmn-eng"  # 資料存放的位置
        self.store_result_path = "../result"  # 检验结果存放的位置