import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocess import padding_mask, sequence_mask


class ScaledDotProductAttention(nn.Module):
    """Scaled dot-product attention mechanism."""

    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, scale=None, attn_mask=None):
        """前向传播.

        Args:
            q: Queries张量，形状为[B, L_q, D_q]
            k: Keys张量，形状为[B, L_k, D_k]
            v: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            scale: 缩放因子，一个浮点标量
            attn_mask: Masking张量，形状为[B, L_q, L_k]

        Returns:
            上下文张量和attetention张量
        """
        """
        torch.bmm(input, mat2, out=None) → Tensor
        Performs a batch matrix-matrix product of matrices stored in input and mat2.
        input and mat2 must be 3-D tensors each containing the same number of matrices.
        If input is a (b \times n \times m)(b×n×m) tensor, mat2 is a (b \times m \times p)(b×m×p) tensor, 
        out will be a (b \times n \times p)(b×n×p) tensor.
        """
        attention = torch.matmul(q, k.transpose(-1,-2))
        # attention [B, L_q, L_k]
        if scale:
            attention = attention * scale
        if attn_mask is not None:
            # 给需要mask的地方设置一个负无穷
            attention = attention.masked_fill_(attn_mask, -np.inf)
        # 计算softmax
        attention = self.softmax(attention)   # attention(..., seq_len_q, seq_len_k)

        # 和V做点积
        context = torch.matmul(attention, v)  # context(..., seq_len_q, depth_v)
        return context, attention


class MultiHeadAttention(nn.Module):

    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()

        self.model_dim = model_dim
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        # multi-head attention之后需要做layer norm
        self.layer_norm = nn.LayerNorm(model_dim)

    def split_heads(self, x, batch_size):
        """
            Split the last dimension into (num_heads, depth).
            Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = x.view(batch_size, -1, self.num_heads, self.dim_per_head)
        return x.permute([0, 2, 1, 3])

    def forward(self, key, value, query, attn_mask=None):
        """
        Args:
            key: Keys张量，形状为[B, L_k, D_k]
            value: Values张量，形状为[B, L_v, D_v]，一般来说就是k
            query: Queries张量，形状为[B, L_q, D_q]
            attn_mask: 形状为[B, L_q, L_k]
        """
        # 残差连接
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads
        query = self.split_heads(query, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        key = self.split_heads(key, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        value = self.split_heads(value, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # if attn_mask is not None:
        #     attn_mask = attn_mask.repeat(num_heads, 1, 1)
        # scaled dot product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(
            query, key, value, scale,
            attn_mask)
        # context[batch_size * num_heads, seq_len, dim_per_head]
        # attention[batch_size * num_heads, L_q, L_k]

        # concat heads
        context = context.permute([0, 2, 1, 3])
        context = context.reshape(batch_size, -1, self.model_dim)
        # context[batch_size, seq_len_q, model_dim]

        # final linear projection
        output = self.linear_final(context)

        # dropout
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len):
        """初始化。

        Args:
            d_model: 一个标量。模型的维度，论文默认是512
            max_seq_len: 一个标量。文本序列的最大长度
        """
        super(PositionalEncoding, self).__init__()

        # 根据论文给的公式，构造出PE矩阵
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # 偶数列使用sin，奇数列使用cos
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # 在PE矩阵的第一行，加上一行全是0的向量，代表这`PAD`的positional encoding
        # 在word embedding中也经常会加上`UNK`，代表位置单词的word embedding，两者十分类似
        # 那么为什么需要这个额外的PAD的编码呢？很简单，因为文本序列的长度不一，我们需要对齐，
        # 短的序列我们使用0在结尾补全，我们也需要这些补全位置的编码，也就是`PAD`对应的位置编码
        pad_row = torch.zeros([1, d_model])  # [1, max_seq_len]
        position_encoding = torch.cat((pad_row, torch.tensor(position_encoding, dtype=torch.float)), dim=0)

        # 嵌入操作，+1是因为增加了`PAD`这个补全位置的编码，
        # Word embedding中如果词典增加`UNK`，我们也需要+1。看吧，两者十分相似
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding,
                                                     requires_grad=False)

    def forward(self, input_len):
        """神经网络的前向传播。
        Args:
          input_len: 一个张量，形状为[BATCH_SIZE, 1]。每一个张量的值代表这一批文本序列中对应的长度。

        Returns:
          返回这一批序列的位置编码，进行了对齐。
        """

        # 找出这一批序列的最大长度
        max_len = torch.max(input_len)
        tensor = torch.cuda.LongTensor if input_len.is_cuda else torch.LongTensor
        # 对每一个序列的位置进行对齐，在原序列位置的后面补上0
        # 这里range从1开始也是因为要避开PAD(0)的位置
        input_pos = tensor(
            [list(range(1, len + 1)) + [0] * (max_len - len) for len in input_len])
        return self.position_encoding(input_pos)


# embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
# # 获得输入的词嵌入编码
# seq_embedding = seq_embedding(inputs)*np.sqrt(d_model)


class PositionalWiseFeedForward(nn.Module):

    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        # x[batch_size, seq_len, model_dim]
        output = x.transpose(1, 2)
        # output[batch_size, model_dim, seq_len]
        output = self.w1(output)  # output[batch_size, ffn_dim, seq_len]
        output = F.relu(output)  # output[batch_size, model_dim, seq_len]
        output = self.w2(output)
        output = self.dropout(output.transpose(1, 2))

        # add residual and norm layer
        output = self.layer_norm(x + output)
        return output


class EncoderLayer(nn.Module):
    """Encoder的一层。"""

    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        # self attention
        context, attention = self.attention(inputs, inputs, inputs,
                                            attn_mask)
        # context[batch_size, seq_len, model_dim]
        # attention[batch_size, seq_len, seq_len]

        # feed forward network
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    """多层EncoderLayer组成Encoder。
    Step:
        Embedding(vocab_size, emb_dim)
        PositionalEncoding(emb_dim, max_seq_len)
        Padding Mask
        encoder_layers(output, self_attention_mask)
    """

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Encoder, self).__init__()

        self.encoder_layers = nn.ModuleList(
            [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])

        # self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)

        return output, attentions


class DecoderLayer(nn.Module):

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()

        self.attention1 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.attention2 = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self,
                dec_inputs,
                enc_outputs,
                self_attn_mask=None,
                context_attn_mask=None):
        # 所有 sub-layers 的主要輸出皆為 (batch_size, target_seq_len, d_model)
        # enc_output 為 Encoder 輸出序列，shape 為 (batch_size, input_seq_len, d_model)
        # self_attention 則為 (batch_size, num_heads, target_seq_len, target_seq_len)
        # context_attention 則為 (batch_size, num_heads, target_seq_len, input_seq_len)
        dec_output, self_attention = self.attention1(
            dec_inputs, dec_inputs, dec_inputs, self_attn_mask)

        # context attention
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention2(
            enc_outputs, enc_outputs, dec_output, context_attn_mask)

        # decoder's output, or context
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):
    """多层DecoderLayer组成Decoder。
    Step:
        Embedding(vocab_size, emb_dim)
        PositionalEncoding(emb_dim, max_seq_len)
        Padding Mask + Sequence Mask
        encoder_layers(output, self_attention_mask)
    """

    def __init__(self,
                 vocab_size,
                 max_seq_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.0):
        super(Decoder, self).__init__()

        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in
             range(num_layers)])

        # self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.seq_embedding = nn.Embedding(vocab_size, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        """
        Args:
            inputs: target sentences[batch_size, target_seq_len]
            inputs_len: target sentences len
            enc_output: encoder output[batch_size, source_seq_len, model_dim]
            context_attn_mask: [B, 1, 1, src_seq_len]
        """
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)

        self_attention_padding_mask = padding_mask(inputs, inputs)  # [batch_size, target_seq_len, target_seq_len]
        seq_mask = sequence_mask(inputs).to(output.device)  # [batch_size, target_seq_len, target_seq_len]
        self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)
        # self_attn_mask[batch_size, target_seq_len, target_seq_len]

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(
                output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)

        return output, self_attentions, context_attentions


class Transformer(nn.Module):
    """
    Args:
        src_vocab_size: 待翻译语言的语料库词汇数
        src_max_len: 待翻译句子的最大长度
        tgt_vocab_size: 目标语言的语料库词汇数
        tgt_max_len: 目标句子的最大长度
        num_layers=6: Encoder和Decoder层数
        model_dim=512: 表示一个词汇的Embedding Dimension
        num_heads=8: MultiHeadAttention的num_heads数目
        ffn_dim=2048: Feed Forward Network的中间维度
        dropout=0.2: Dropout概率
    """

    def __init__(self,
                 src_vocab_size,
                 src_max_len,
                 tgt_vocab_size,
                 tgt_max_len,
                 num_layers=6,
                 model_dim=512,
                 num_heads=8,
                 ffn_dim=2048,
                 dropout=0.2):
        super(Transformer, self).__init__()
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim,
                               num_heads, ffn_dim, dropout)

        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(src_seq, tgt_seq)  # shape [B, 1, 1, src_seq_len]

        output, enc_self_attn = self.encoder(src_seq, src_len)  # output[batch_size, src_len, model_dim]

        output, dec_self_attn, ctx_attn = self.decoder(
            tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        # output = self.softmax(output)
        # output[batch_size, seq_len, tgt_vocab_size]
        # enc_self_attn[num_layers, model_dim, seq_len, seq_len]
        # dec_self_attn[num_layers, model_dim, seq_len, seq_len]
        # ctx_attn[num_layers, model_dim, seq_len, seq_len]
        return output, enc_self_attn, dec_self_attn, ctx_attn
