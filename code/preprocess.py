import torch
from nltk.translate.bleu_score import sentence_bleu


def padding_mask(seq_k, seq_q):
    # seq_k和seq_q的形状都是[B,L]
    len_q = seq_q.size(1)
    # `PAD` is 1
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).unsqueeze(1)  # shape [B, 1, 1, L_k]
    return pad_mask


def sequence_mask(seq):
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8),
                      diagonal=1)
    # mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # shape [B, 1, 1, L_k]
    return mask


def tokens2sentence(outputs, int2word):
    """## 數字轉句子"""
    sentences = []
    for tokens in outputs:
        sentence = []
        for token in tokens:
            word = int2word[str(int(token))]
            if word == '<EOS>':
                break
            sentence.append(word)
        sentences.append(sentence)

    return sentences


def compute_bleu(sentences, targets):
    """計算 BLEU score"""
    score = 0
    assert (len(sentences) == len(targets))

    def cut_token(sentence):
        tmp = []
        for token in sentence:
            if token == '<UNK>' or token.isdigit() or len(bytes(token[0], encoding='utf-8')) == 1:
                tmp.append(token)
            else:
                tmp += [word for word in token]
        return tmp

    for sentence, target in zip(sentences, targets):
        sentence = cut_token(sentence)
        target = cut_token(target)
        score += sentence_bleu([target], sentence, weights=(1, 0, 0, 0))

    return score


def infinite_iter(data_loader):
    """##迭代 dataloader"""
    it = iter(data_loader)
    while True:
        try:
            ret = next(it)
            yield ret
        except StopIteration:
            it = iter(data_loader)
