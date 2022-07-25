import torch
import torch.nn as nn
from models.attn import *
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SelfConvergence(nn.Module):
    '''Smoothing Filter Attention Mechanism'''
    def __init__(self, seq_len, val_dim, self_masked):
        super(SelfConvergence, self).__init__()
        self.weight = nn.Parameter(torch.rand(seq_len, val_dim))
        self.weight = nn.init.uniform_(self.weight, a=0, b=0.4)
        self.seq_len = seq_len
        self.val_dim = val_dim
        self.masked = self_masked
    def forward(self,x, inf=1e4):
        '''
        x : batch_size, seq_len, val_dim
        '''
        #x1: batch_size, seq_len, seq_len, val_dim
        #x2: batch_size, seq_len, seq_len, val_dim
        #w:  batch_size, seq_len, seq_len, val_dim
        batch_size = x.size(0)
        key = x.unsqueeze(dim=1).expand(-1, self.seq_len, -1, -1)
        query = x.unsqueeze(dim=1).expand(-1, self.seq_len, -1, -1).transpose(1,2)
        w = self.weight.unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, self.seq_len, -1, -1)
        if self.masked:
            eyes = (torch.eye(self.seq_len, self.seq_len)*inf).unsqueeze(0).unsqueeze(3).expand(batch_size, -1, -1, self.val_dim).to(device)
            result = torch.softmax(-((key-query+eyes)**2)*w, dim=1)*query
        else:
            result = torch.softmax(-((key-query)**2)*w, dim=1)*query
        return result.sum(dim=1), x


class DeepthDifferenceBlock():
    '''DepthDifference and Inverse DD'''
    def __init__(self):
        super(DeepthDifferenceBlock, self).__init__()
    def __call__(self, x):
        '''
        x: [batch_size, seq_len, val_dim
        '''
        batch_size, val_dim = x.size(0), x.size(2)
        x = x.transpose(1,2)
        pad = (2*x[:, :,-1]-x[:, :,-2])
        pad = pad.view(-1, x.size(1), 1)
        x_pad = torch.cat([x, pad], dim=2)

        diff_disloc = x_pad[:,:,1:]
        diff_oriloc = x_pad[:,:,:-1]
        diff = diff_disloc-diff_oriloc
        original_data = x_pad[:,:,0].view(batch_size, 1, val_dim)
        diff = diff.transpose(1,2)
        return diff, original_data

class InverseDeepthDifferenceBlock():
    def __init__(self):
        super(InverseDeepthDifferenceBlock, self).__init__()
    def __call__(self, dif, ori):
        '''
        diff: [batch_size, seq_len, val_dim]
        origial: [batch_size, 1, val_dim]
        '''
        diffNoLast = dif[:, :-1, :].cumsum(dim=1)
        cumsum = diffNoLast+ori
        inverse_x = torch.cat([ori, cumsum], dim=1)
        return inverse_x