import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
import re
import os

from config import Config
from data import EN2CNDataset
from preprocess import infinite_iter, tokens2sentence
from model import Transformer
from test import evaluate

if __name__ == '__main__':
    config = Config()

    """## 訓練流程
    - 先訓練，再檢驗
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
    total_steps = 0
    if config.load_model:
        transformer_model.load_state_dict(torch.load(config.load_model_path))
        total_steps = int(re.split('[_/.]', config.model_file)[1])

    optimizer = torch.optim.Adam(transformer_model.parameters(), lr=config.learning_rate)
    loss_function = CrossEntropyLoss(ignore_index=0)

    train_losses, val_losses, bleu_scores = [], [], []

    while total_steps < config.num_steps:
        # 訓練模型
        transformer_model.train()
        transformer_model.zero_grad()
        losses = []
        loss_sum = 0.0
        for step in range(config.summary_steps):
            source, target = next(train_iter)  # sources targets[batch_size, max_output_len]
            source, target = source.to(config.device), target.to(config.device)
            target_input, target_real = target[:, :-1], target[:, 1:]

            src_len = torch.tensor([source.shape[1] for _ in range(source.shape[0])]).to(config.device)
            tgt_len = torch.tensor([target_input.shape[1] for _ in range(source.shape[0])]).to(config.device)
            output, enc_self_attn, dec_self_attn, ctx_attn = transformer_model(source, src_len,
                                                                               target_input, tgt_len)

            if config.DEBUG:
                pred_idx = output.argmax(axis=-1).view(source.shape[0], -1)
                pred_sentences = tokens2sentence(pred_idx[:, 1:], train_dataset.int2word_cn)
                src_sentences = tokens2sentence(source[:, 1:], train_dataset.int2word_en)

            # output = [batch size, seq len, vocab size]

            # # targets 的第一個 token 是 <BOS> 所以忽略
            # output = output[:, 1:].reshape(-1, output.size(2))
            # # output = [batch_size * (output_len - 1), target_vocab_size]
            # target = target[:, 1:].reshape(-1)
            # # targets = [batch_size * (target_len - 1)]
            output = output.reshape(-1, output.size(2))
            target_real = target_real.reshape(-1)
            loss = loss_function(output, target_real)

            optimizer.zero_grad()
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(transformer_model.parameters(), 1)
            optimizer.step()

            loss_sum += loss.item()
            if (step + 1) % 10 == 0:
                loss_sum = loss_sum / 10
                # print("\r", "train [{}] loss: {:.3f}, Perplexity: {:.3f}".format(total_steps + step + 1, loss_sum,
                #                                                                  np.exp(loss_sum)), end=" ")
                print("train [{}] loss: {:.3f}, Perplexity: {:.3f}".format(total_steps + step + 1, loss_sum,
                                                                                 np.exp(loss_sum)))
                losses.append(loss_sum)
                loss_sum = 0.0
        train_losses += losses
        total_steps += config.summary_steps

        # 儲存模型和結果
        if total_steps % config.store_steps == 0 or total_steps >= config.num_steps:
            torch.save(transformer_model.state_dict(), f'{config.store_model_path}/model_{total_steps}.ckpt')
            # with open(f'{config.store_model_path}/output_{total_steps}.txt', 'w') as f:
            #   for line in result:
            #     print (line, file=f)

        # 檢驗模型
        if total_steps > 3000 or total_steps == config.summary_steps:
            val_loss, bleu_score, result = evaluate(transformer_model, val_loader, loss_function)
            val_losses.append(val_loss)
            bleu_scores.append(bleu_score)

            print("\r", "val [{}] loss: {:.3f}, Perplexity: {:.3f}, blue score: {:.3f}".format(total_steps, val_loss,
                                                                                               np.exp(val_loss),
                                                                                               bleu_score))
            print("\r", "val [{}]".format(total_steps))

            # 儲存結果
            result_file = f'model_{total_steps}.txt'
            with open(os.path.join(config.store_result_path, result_file), 'w') as f:
                for line in result:
                    print(line, file=f)

