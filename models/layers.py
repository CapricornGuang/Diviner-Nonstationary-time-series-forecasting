import torch
import torch.nn as nn
from models.blocks import *
from models.attn import *
import models.utils as utils
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderLayer(torch.nn.Module):
    def __init__(self, seq_len, dim_val, dim_attn, n_heads, sc, ddblock, self_masked):
        super(EncoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.ddblock = ddblock
        if ddblock:
            self.dd = DeepthDifferenceBlock()
            self.idd = InverseDeepthDifferenceBlock()
        if sc:
            self.conver = SelfConvergence(seq_len, dim_val, self_masked)
        self.sc = sc
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
    def forward(self, x):
        #x: stands for trend. [batch_size, seq_len, dim_val]
        
        #self-convergence block
        if self.sc:
            cx, orix = self.conver(x)
        else:
            cx, orix = x, x
            
        #self-attention and dd block
        if self.ddblock:
            dx, oridata = self.dd(cx)
            ax = self.attn(dx)
            ax = self.idd(ax, oridata)
        else:
            ax = self.attn(cx)
        rx = ax+cx
        
        #feed forward
        lx = self.fc1(torch.relu(self.fc2(rx)))
        rx = lx+rx
        return rx, orix, cx

class DecoderLayer(torch.nn.Module):
    def __init__(self, seq_len, dim_val, dim_attn, n_heads, sc, ddblock, self_masked):
        super(DecoderLayer, self).__init__()
        self.attn = MultiHeadAttentionBlock(dim_val, dim_attn, n_heads)
        self.ddblock = ddblock
        if ddblock:
            self.dd = DeepthDifferenceBlock()
            self.idd = InverseDeepthDifferenceBlock()
        if sc:
            self.conver = SelfConvergence(seq_len, dim_val, self_masked)
        self.sc = sc
        self.fc1 = nn.Linear(dim_val, dim_val)
        self.fc2 = nn.Linear(dim_val, dim_val)
        
    def forward(self, x, enc):
        #x: stands for trend. batch_size, seq_len, dim_val
        #self-convergence block
        if self.sc:
            cx, orix = self.conver(x)
        else:
            cx, orix = x, x
            
        #self-attention and DD block
        if self.ddblock:
            dx, oridata = self.dd(cx)
            encx, _ = self.dd(enc)
            ax = self.attn(dx, kv=encx)
            ax = self.idd(ax, oridata)
        else:
            ax = self.attn(cx, kv=enc)
        rx = ax+cx
        
        #feed forward
        lx = self.fc1(torch.relu(self.fc2(rx)))
        rx = lx+rx
        return rx, orix, cx


class GenerativeOutputLayer(torch.nn.Module):
    def __init__(self, dec_seq_len, out_seq_len, dim_val, output_size, kernel, padding,layers):
        super(GenerativeOutputLayer, self).__init__()
        self.layers = layers
        self.dim_val = dim_val
        self.output_size = output_size
        
        self.outs = utils.get_module_list(nn.Conv2d(dec_seq_len, out_seq_len, kernel, padding=padding), 1)
        self.outs = utils.get_module_list(nn.Conv2d(out_seq_len, out_seq_len, kernel, padding=padding), layers-1, lists=self.outs)

        if self.dim_val != self.output_size:
            self.lnr = nn.Linear(dim_val, output_size)

    def forward(self,x):
        '''
        x: [batch_size, seq_len, val_dim]
        '''
        #x: [batch_size, seq_len, val_dim] -> [batch_size, seq_len, val_dim, 1]
        if self.dim_val != self.output_size:
            x = self.lnr(x)
        x = x.unsqueeze(3)

        x = self.outs[0](x)
        if self.layers >1:
            for out in self.outs[1:]:
                x = F.elu(out(x))+x
        x = x.squeeze(3)
        
        return x