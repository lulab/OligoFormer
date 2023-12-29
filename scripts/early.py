import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from collections import OrderedDict
import math
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, average_precision_score, roc_curve,auc,precision_recall_curve
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Conv1dReLU(nn.Module):
    '''
    kernel_size=3, stride=1, padding=1
    kernel_size=5, stride=1, padding=2
    kernel_size=7, stride=1, padding=3
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU()
        )
    def forward(self, x):
        return self.inc(x)

class StackCNN(nn.Module):
    def __init__(self, layer_num, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.inc = nn.Sequential(OrderedDict([('conv_layer0', Conv1dReLU(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))]))
        for layer_idx in range(layer_num - 1):
            self.inc.add_module('conv_layer%d' % (layer_idx + 1), Conv1dReLU(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
        self.inc.add_module('pool_layer', nn.AdaptiveMaxPool1d(1))
    def forward(self, x):
        return self.inc(x).squeeze(-1)

class siRNA_Encoder2(nn.Module):
    def __init__(self, block_num, vocab_size, embedding_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.block_list = nn.ModuleList()
        for block_idx in range(block_num):
            self.block_list.append(
                StackCNN(block_idx+1, embedding_dim, 96, 3)
            )
        self.linear = nn.Linear(block_num * 96, 96)
    def forward(self, x):
        x = self.embed(x).permute(0, 2, 1)
        feats = [block(x) for block in self.block_list]
        x = torch.cat(feats, -1)
        x = self.linear(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False
        pos = torch.arange(0, max_len, device=device)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position
        _2i = torch.arange(0, d_model, step=2, device=device).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)
        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        # [batch_size = 128, seq_len = 30]
        return self.encoding[:seq_len, :]

class TokenEmbedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):
        super(TokenEmbedding, self).__init__(vocab_size, d_model, padding_idx=1)

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PositionalEncoding(d_model, max_len)
    def forward(self, x):
        # tok_emb = self.tok_emb(x)
        tok_emb = x
        pos_emb = self.pos_emb(x)
        return tok_emb + pos_emb

class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, q, k, v, mask=None, e=1e-12):
        batch_size, head, length, d_tensor = k.size()
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)
        score = self.softmax(score)
        v = score @ v
        return v, score

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, unbiased=False, keepdim=True)
        out = (x - mean) / torch.sqrt(var + self.eps)
        out = self.gamma * out + self.beta
        return out

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)
    def forward(self, q, k, v, mask=None):
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        q, k, v = self.split(q), self.split(k), self.split(v)
        out, attention = self.attention(q, k, v, mask=mask)
        out = self.concat(out)
        out = self.w_concat(out)
        return out
    def split(self, tensor):
        batch_size, length, d_model = tensor.size()
        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        return tensor
    def concat(self, tensor):
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor
        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor

class EncoderLayer(nn.Module):
    def __init__(self, d_model, ffn_hidden, n_head):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout = nn.Dropout(0.02)
        self.ffn = PositionwiseFeedForward(d_model=d_model, hidden=ffn_hidden)
        self.norm2 = LayerNorm(d_model=d_model)
    def forward(self, embed, src_mask):
        attn = self.attention(q=embed, k=embed, v=embed, mask=src_mask) # 16,19,64
        attn = self.dropout(attn)
        ln1 = self.norm1(attn) # + embed
        ff = self.ffn(ln1)
        ln2 = self.norm2(ff + ln1)
        ln2 = self.dropout(ln2)
        return attn

class Encoder(nn.Module):
    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers):
        super().__init__()
        self.emb = TransformerEmbedding(d_model=d_model,max_len=max_len,vocab_size=enc_voc_size)
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,ffn_hidden=ffn_hidden,n_head=n_head) for _ in range(n_layers)])
    def forward(self, x, src_mask):
        x = self.emb(x) # 16，19，64
        for layer in self.layers:
            x = layer(x, src_mask)
        return x # 16，19，64

class RNA_Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, lstm_dim, n_head, n_layers):
        super().__init__()
        self.conv2d_1 = nn.Conv2d(1, 16, kernel_size=(5,5), stride=1, padding=0)
        self.conv2d_2 = nn.Conv2d(16, 64, kernel_size=(5,5), stride=1, padding=0)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.02)
        self.avgpool = nn.AvgPool2d((2,2))
        self.maxpool = nn.MaxPool2d((2,2))
        self.biLSTM = nn.LSTM(4,lstm_dim,2,bidirectional = True) # 68
        self.encoder = Encoder(vocab_size,64,lstm_dim*2,embedding_dim,n_head,n_layers)

    def forward(self, x):
        x = x.float() # 16,1,19,20
        x = self.conv2d_1(x) # 16,64,17,18
        x = self.relu(x) # 16,64,17,18
        x = self.maxpool(x)
        x = self.conv2d_2(x) # 16,64,3,4
        x = self.relu(x) # 16,64,19,1
        x1 = self.avgpool(x) # 16,19,32
        x2 = self.maxpool(x) # 16,19,32
        x = torch.cat([x1,x2],dim = -1) # 16,64,1,4
        x = x.squeeze()
        x, _ = self.biLSTM(x) # 16,19,64
        x = self.relu(x) # 16,19,64
        x = self.dropout(x) # 16,19,64 #######
        x = self.encoder(x,src_mask = None) # 16,19,64
        return x



class Oligo_early(nn.Module):
    def __init__(self, vocab_protein_size = 26, embedding_dim = 128, lstm_dim = 32,  n_head = 8, n_layers = 1): # 4928
        super().__init__()
        self.RNA_encoder = RNA_Encoder(vocab_protein_size, embedding_dim, lstm_dim, n_head, n_layers)
        self.RNA_avgpool = nn.AvgPool2d((19 * 4, 5))
        self.classifier = nn.Sequential(
            nn.Linear(4224, 256),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.02),
            nn.Linear(64, 2),
            nn.Softmax()
        )
        self.flatten = nn.Flatten()
    def forward(self, siRNA, mRNA, siRNA_FM, mRNA_FM):
        mRNA = mRNA.view((mRNA.shape[0], mRNA.shape[1], mRNA.shape[2] // 3, mRNA.shape[3] * 3))
        RNA = torch.cat([siRNA,mRNA],dim = -1) # 4608
        RNA = self.RNA_encoder(RNA)
        RNA = self.flatten(RNA)
        RNA_FM = torch.cat([siRNA_FM,mRNA_FM],dim = 1) 
        RNA_FM = self.RNA_avgpool(RNA_FM)
        RNA_FM = RNA_FM.view(RNA_FM.shape[0], RNA_FM.shape[2])
        RNA_FM = self.flatten(RNA_FM)
        merge = torch.cat([RNA,RNA_FM],dim = -1) # 4864
        # merge = torch.cat([siRNA,mRNA],dim = -1) # 4608
        # merge = torch.cat([siRNA_FM,mRNA_FM],dim = -1) # 256
        x = self.classifier(merge)
        return x

