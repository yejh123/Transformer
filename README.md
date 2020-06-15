# Transformer：Seq2Seq 模型 + 自注意力機制
  - [neural-machine-translation-with-transformer-and-tensorflow2](https://leemeng.tw/neural-machine-translation-with-transformer-and-tensorflow2.html)
  - [Transformer 模型的 PyTorch 实现](https://juejin.im/post/5b9f1af0e51d450e425eb32d)
  - [Google 在 2017 年 6 月的一篇論文：Attention Is All You Need](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)
  - [Google AI Blog: transformer-novel-neural-network](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
  - [ HarvardNLP 以 Pytorch 實現的 The Annotated Transformer](http://nlp.seas.harvard.edu//2018/04/03/attention.html#additional-components-bpe-search-averaging)
  - [进击的BERT](https://leemeng.tw/attack_on_bert_transfer_learning_in_nlp.html)
  - [OpenAI 的 GPT](https://openai.com/blog/better-language-models/)
  - [机器翻译自动评估-BLEU算法详解](https://blog.csdn.net/qq_31584157/article/details/77709454)
  - [Batch Normalization](https://www.cnblogs.com/shine-lee/p/11989612.html)
  - [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
  
## 概念
循環神經網路 RNN 時常被拿來處理序列數據，但其運作方式存在著一個困擾研究者已久的問題：無法有效地平行運算。

![自注意層可以做到跟雙向 RNN 一樣的事情，還可以平行運算](https://leemeng.tw/images/transformer/rnn-vs-self-attn-layer.jpg)

Google 在 2017 年 6 月的一篇論文：Attention Is All You Need 裡參考了注意力機制，提出了自注意力機制（Self-Attention mechanism）。這個機制不只跟 RNN 一樣可以處理序列數據，還可以平行運算。

一個自注意層（Self-Attention Layer）可以利用矩陣運算在等同於 RNN 的一個時間點內就回傳所有 bi ，且每個 bi 都包含了整個輸入序列的資訊。相比之下，RNN 得經過 4 個時間點依序看過 [a1, a2, a3, a4] 以後才能取得序列中最後一個元素的輸出 b4 。

雖然我們一直強調自注意力機制的平行能力，如果你還記得我們在上一節講述的注意力機制，就會發現在 Seq2Seq 架構裡頭自注意力機制跟注意力機制講的根本是同樣一件事情：

  - 注意力機制讓 Decoder 在生成輸出元素的 repr. 時關注 Encoder 的輸出序列，從中獲得上下文資訊
  - 自注意力機制讓 Encoder 在生成輸入元素的 repr. 時關注自己序列中的其他元素，從中獲得上下文資訊
  - 自注意力機制讓 Decoder 在生成輸出元素的 repr. 時關注自己序列中的其他元素，從中獲得上下文資訊

總之透過新設計的自注意力機制以及原有的注意力機制，Attention Is All You Need 論文作者們打造了一個完全不需使用 RNN 的 Seq2Seq 模型：Transformer。以下是 Transformer 中非常簡化的 Encoder-Decoder 版本，讓我們找找哪邊用到了（自）注意力機制：

![在 Transformer 裡頭共有 3 個地方用到（自）注意力機制](https://leemeng.tw/images/transformer/Transformer_decoder.png)

在 Transformer 裡頭，Decoder 利用注意力機制關注 Encoder 的輸出序列（Encoder-Decoder Attention），而 Encoder 跟 Decoder 各自利用自注意力機制關注自己處理的序列（Self-Attention）。無法平行運算的 RNN 完全消失，名符其實的 Attention is all you need.

以 Transformer 實作的 NMT 系統基本上可以分為 6 個步驟：
  - Encoder 為輸入序列裡的每個詞彙產生初始的 repr. （即詞向量），以空圈表示
  - 利用自注意力機制將序列中所有詞彙的語義資訊各自匯總成每個詞彙的 repr.，以實圈表示
  - Encoder 重複 N 次自注意力機制，讓每個詞彙的 repr. 彼此持續修正以完整納入上下文語義
  - Decoder 在生成每個法文字時也運用了自注意力機制，關注自己之前已生成的元素，將其語義也納入之後生成的元素
  - 在自注意力機制後，Decoder 接著利用注意力機制關注 Encoder 的所有輸出並將其資訊納入當前生成元素的 repr.
  - Decoder 重複步驟 4, 5 以讓當前元素完整包含整體語義

## 应用
  - 文本摘要（Text Summarization）
  - 圖像描述（Image Captioning）
  - 閱讀理解（Reading Comprehension）
  - 語音辨識（Voice Recognition）
  - 語言模型（Language Model）
  - 聊天機器人（Chat Bot）
  - 其他任何可以用 RNN 的潛在應用

## 模型架构
![Transformer架构](https://leemeng.tw/theme/images/left-nav/transformer.jpg)
### Transformer
  - Encoder
    - 輸入 Embedding
    - 位置 Encoding
    - N 個 Encoder layers
      - sub-layer 1: Encoder 自注意力機制
      - sub-layer 2: Feed Forward
  - Decoder
    - 輸出 Embedding
    - 位置 Encoding
    - N 個 Decoder layers
      - sub-layer 1: Decoder 自注意力機制
      - sub-layer 2: Decoder-Encoder 注意力機制
      - sub-layer 3: Feed Forward
  - Final Dense Layer
  
### Word Embedding
词向量模型

### Self-Attention layer
  - 將 q 和 k 做點積得到 matmul_qk
  - 將 matmul_qk 除以 scaling factor sqrt(dk)
  - 有遮罩的話在丟入 softmax 前套用
  - 通過 softmax 取得加總為 1 的注意權重
  - 以該權重加權平均 v 作為輸出結果
  - 回傳輸出結果以及注意權重
  
#### Scaled dot-product attention：一种注意函数
![Scaled dot-product attention](https://leemeng.tw/images/transformer/scaled-dot-product.jpg)

#### Mask 
mask顾名思义就是掩码，在我们这里的意思大概就是对某些值进行掩盖，使其不产生效果。

需要说明的是，我们的Transformer模型里面涉及两种mask。分别是padding mask和look ahead mask（也称作sequence mask）。其中后者我们已经在decoder的self-attention里面见过啦！

其中，padding mask在所有的scaled dot-product attention里面都需要用到，而look ahead mask只有在decoder的self-attention里面用到。

  - padding mask：遮住 <pad> token 不讓所有子詞關注
  - look ahead mask：遮住 Decoder 未來生成的子詞不讓之前的子詞關注

#### Multi-head attention
mutli-head attention 的概念本身並不難，用比較正式的說法就是將 Q、K 以及 V 這三個張量先個別轉換到 d_model 維空間，再將其拆成多個比較低維的 depth 維度 N 次以後，將這些產生的小 q、小 k 以及小 v 分別丟入前面的注意函式得到 N 個結果。接著將這 N 個 heads 的結果串接起來，最後通過一個線性轉換就能得到 multi-head attention 的輸出。
<br/>
而為何要那麼「搞剛」把本來 d_model 維的空間投影到多個維度較小的子空間（subspace）以後才各自進行注意力機制呢？這是因為這給予模型更大的彈性，讓它可以同時關注不同位置的子詞在不同子空間下的 representation，而不只是本來 d_model 維度下的一個 representation。

#### Residual Connection
假设网络中某个层对输入x作用后的输出是F(x)，那么增加residual connection之后，就变成了：F(x) + x

这个+x操作就是一个shortcut。

那么残差结构有什么好处呢？显而易见：因为增加了一项x，那么该层网络对x求偏导的时候，多了一个常数项1！所以在反向传播过程中，梯度连乘，也不会造成梯度消失！

#### Layer Normalization
Normalization有很多种，但是它们都有一个共同的目的，那就是把输入转化成均值为0方差为1的数据。我们在把数据送入激活函数之前进行normalization（归一化），因为我们不希望输入数据落在激活函数的饱和区。

那么什么是Layer normalization呢？:它也是归一化数据的一种方式，不过LN是在每一个样本上计算均值和方差，而不是BN那种在批方向计算均值和方差！


#### Positional Encoding
透過多層的自注意力層，Transformer 在處理序列時裡頭所有子詞都是「天涯若比鄰」：想要關注序列中任何位置的資訊只要 O(1) 就能辦到。這讓 Transformer 能很好地 model 序列中長距離的依賴關係（long-range dependencise）。但反過來說 Transformer 則無法 model 序列中字詞的順序關係，所以我們得額外加入一些「位置資訊」給 Transformer。

這個資訊被稱作位置編碼（Positional Encoding），實作上是直接加到最一開始的英文 / 中文詞嵌入向量（word embedding）裡頭。其直觀的想法是想辦法讓被加入位置編碼的 word embedding 在 d_model 維度的空間裡頭不只會因為語義相近而靠近，也會因為位置靠近而在該空間裡頭靠近。

![位置编码公式](https://leemeng.tw/images/transformer/position-encoding-equation.jpg)

論文裡頭提到他們之所以這樣設計位置編碼（Positional Encoding, PE）是因為這個函數有個很好的特性：給定任一位置 pos 的位置編碼 PE(pos)，跟它距離 k 個單位的位置 pos + k 的位置編碼 PE(pos + k) 可以表示為 PE(pos) 的一個線性函數（linear function）。

### Context-attention

## Training
### Teacher Forcing

### 定义损失函数与指标
  - cross entropy
  - train loss
  - train accuracy
  
### 设置超参数
  - num_layers 決定 Transfomer 裡頭要有幾個 Encoder / Decoder layers
  - d_model 決定我們子詞的 representation space 維度
  - num_heads 要做幾頭的自注意力運算
  - dff 決定 FFN 的中間維度
  - dropout_rate 預設 0.1，一般用預設值即可
  - input_vocab_size：輸入語言（英文）的字典大小
  - target_vocab_size：輸出語言（中文）的字典大小
  
### 设置Optimizer

### 定时存档


## Testing
利用 Transformer 進行翻譯（預測）的邏輯如下：

  - 將輸入的英文句子利用 Subword Tokenizer 轉換成子詞索引序列（還記得 inp 吧？）
  - 在該英文索引序列前後加上代表英文 BOS / EOS 的 tokens
  - 在 Transformer 輸出序列長度達到 MAX_LENGTH 之前重複以下步驟：
    - 為目前已經生成的中文索引序列產生新的遮罩
    - 將剛剛的英文序列、當前的中文序列以及各種遮罩放入 Transformer
    - 將 Transformer 輸出序列的最後一個位置的向量取出，並取 argmax 取得新的預測中文索引
    - 將此索引加到目前的中文索引序列裡頭作為 Transformer 到此為止的輸出結果
    - 如果新生成的中文索引為 <end> 則代表中文翻譯已全部生成完畢，直接回傳
  - 將最後得到的中文索引序列回傳作為翻譯結果
  
  
## 可视化注意权重



