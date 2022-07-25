import torch
import torch.nn as nn
import math
from models.attn import AttentionBlock
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PositionalEncoding(nn.Module):
    '''
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    '''

    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]. squeeze(1)
        return x


class MultiChannelsEmbedding(nn.Module):
    def __init__(self, input_size, token_dim):
        super(MultiChannelsEmbedding, self).__init__()
        self.Attention = AttentionBlock(input_size, token_dim)
    def forward(self, x):
        '''
        We note that the targeted feature should be in the FIRST channel
        x: [batch_size, seq_len, channels, dim]
        ---output---
        embeds: [batch_size, seq_len, dim]
        '''
        
        x = torch.transpose(x, 1, 2) #->[bs, channels, seq_len, dim]
        if x.size(1)==1:
            return x.squeeze(1)
            
        seq_len = x.size(2)
        embeds = []
        for i in range(seq_len):
            #kv: [batch_size, 1, dim]
            #q: [batch_size, seq_len, dim]
            q = x[:,0,i,:].unsqueeze(1)
            kv = x[:,:,i,:]
            embeds.append(self.Attention(q, kv=kv))
        embeds = torch.cat(embeds, dim=1)
        return embeds