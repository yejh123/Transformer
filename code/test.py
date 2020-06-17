import re
import os
import numpy as np
import torch
from torch.nn.functional import pad
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

from config import Config
from preprocess import tokens2sentence, compute_bleu
from data import EN2CNDataset
from model import Transformer
from preprocess import infinite_iter, tokens2sentence

def evaluate(model, dataloader, loss_function):
    model.eval()
    config = Config()
    loss_sum, bleu_score = 0.0, 0.0
    n = 0
    result = []
    for sources, targets in dataloader:
        sources, targets = sources.to(config.device), targets.to(config.device)
        batch_size = sources.size(0)

        # Decoder 在第一個時間點吃進去的輸入是一個只包含一個中文 <BOS> token 的序列
        outputs = torch.tensor([1]).unsqueeze(dim=0).expand(batch_size, -1).to(config.device)  # outputs[batch_size, 1]
        for i in range(config.max_output_len):
            sources_len = torch.tensor([sources.shape[1] for _ in range(sources.shape[0])]).to(config.device)
            output_len = torch.tensor([outputs.shape[1] for _ in range(sources.shape[0])]).to(config.device)
            predictions, _, _, _ = model(sources, sources_len, outputs, output_len)
            # predictions = [batch size, seq len, vocab size]
            # 將序列中最後一個 distribution 取出，並將裡頭值最大的當作模型最新的預測字
            predictions_word = predictions[:, -1, :]
            pred_idx = predictions_word.argmax(axis=-1).view(batch_size, -1)

            outputs = torch.cat((outputs, pred_idx), 1)
            # 遇到 <end> token 就停止回傳，代表模型已經產生完結果
            if pred_idx.item() == 2:
                temp = pad(outputs, pad=[0, config.max_output_len - outputs.shape[1] - 1], value=0)
                predictions, _, _, _ = model(sources, sources_len, temp, sources_len-1)
                break

        predictions = predictions.view(-1, predictions.size(2))
        # targets 的第一個 token 是 <BOS> 所以忽略
        targets = targets[:, 1:].view(-1)

        loss = loss_function(predictions, targets)
        loss_sum += loss.item()

        # 將預測結果轉為文字
        targets = targets.view(batch_size, -1)
        pred_sentences = tokens2sentence(outputs[:, 1:], dataloader.dataset.int2word_cn)
        sources = tokens2sentence(sources, dataloader.dataset.int2word_en)
        targets = tokens2sentence(targets, dataloader.dataset.int2word_cn)
        for source, prediction, target in zip(sources, pred_sentences, targets):
            result.append((source, prediction, target))
        # 計算 Bleu Score
        bleu_score += compute_bleu(pred_sentences, targets)

        n += batch_size
        print("\r", f"test_epoch[{n}]", end=" ")
    return loss_sum / len(dataloader), bleu_score / n, result


if __name__ == '__main__':
    config = Config()

    """## 检验训练结果
    """
    # 準備訓練資料
    print("加载数据中...")
    train_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'training')
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    train_iter = infinite_iter(train_loader)
    # 準備檢驗資料
    val_dataset = EN2CNDataset(config.data_path, config.max_output_len, 'validation')
    val_loader = DataLoader(val_dataset, batch_size=1)
    print("加载数据完成...")
    # 建構模型
    transformer_model = Transformer(train_dataset.en_vocab_size,
                                    config.max_output_len,
                                    train_dataset.cn_vocab_size,
                                    config.max_output_len,
                                    num_layers=config.n_layers,
                                    model_dim=config.model_dim,
                                    num_heads=config.num_heads,
                                    ffn_dim=config.ffn_dim,
                                    dropout=config.dropout,
                                    ).to(config.device)
    print("使用模型：")
    print(transformer_model)
    loss_function = CrossEntropyLoss(ignore_index=0)
    if config.load_model:
        transformer_model.load_state_dict(torch.load(config.load_model_path))
        total_steps = int(re.split('[_/.]', config.model_file)[1])

    # 檢驗模型
    val_losses, bleu_scores = [], []
    val_loss, bleu_score, result = evaluate(transformer_model, val_loader, loss_function)
    val_losses.append(val_loss)
    bleu_scores.append(bleu_score)

    print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}".format(total_steps, val_loss,
                                                                                       np.exp(val_loss),
                                                                                       bleu_score))
    print("\r", "val [{}]".format(total_steps))

    # 儲存結果
    result_file = re.split('[.]', config.model_file)[0] + '.txt'
    with open(os.path.join(config.store_result_path, result_file), 'w') as f:
        for line in result:
            print(line, file=f)