import torch
import torch.nn as nn
import models.utils as utils
from models.blocks import *
from models.layers import *
from models.embed import PositionalEncoding, MultiChannelsEmbedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Diviner(torch.nn.Module):
    def __init__(self, dim_val, dim_attn, dim_attn_channel, dim_input, output_size, enc_seq_len, dec_seq_len, out_seq_len, 
    n_decoder_layers = 1, n_encoder_layers = 1, n_heads = 1, convout={'use':True, 'kernel':5, 'layers':3}, 
    sc=True, ddblock=True, self_masked=True):
        super(Diviner, self).__init__()
        #init parameters
        if convout['use']:
            assert utils.is_odd(convout['kernel']), 'kernel size should be set to be an odd'
            convout['pad'] = utils.get_keepdim_paddings(convout['kernel'])
        self.dec_seq_len = dec_seq_len
        self.output_size = output_size

        #model selection
        self.sc = sc
        self.ddblock = ddblock
        
        #Encoder-Decoder
        if sc:
            self.conver = SelfConvergence(enc_seq_len, dim_val, self_masked)
        self.encs = utils.get_module_list(EncoderLayer(enc_seq_len, dim_val, dim_attn, n_heads, sc, ddblock, self_masked), n_encoder_layers)
        self.decs = utils.get_module_list(DecoderLayer(dec_seq_len, dim_val, dim_attn, n_heads, sc, ddblock, self_masked), n_decoder_layers)
        
        #Input layer
        self.pos = PositionalEncoding(dim_val)
        self.multi_channels = MultiChannelsEmbedding(dim_input, token_dim=dim_attn_channel)
        self.enc_input_fc1 = nn.Linear(dim_input, dim_val)
        
        #Output layer
        self.convout = convout['use']
        self.out_fc1 = GenerativeOutputLayer(dec_seq_len, out_seq_len, dim_val, output_size, 
        kernel=(convout['kernel'],1), 
        padding=(convout['pad'],0),
        layers=convout['layers']) if convout['use'] else nn.Linear(dec_seq_len * dim_val, out_seq_len * output_size)
        
            
    def forward(self, x):
        #Time-series Embeddings
        x = self.multi_channels(x) if len(x.size())==4 else x
        x = self.pos(self.enc_input_fc1(x))

        #Encoder
        smo = []
        ex, orix, cx = self.encs[0](x)
        smo.append((cx,orix))
        for enc in self.encs[1:]:
            ex, orix, cx = enc(ex)
            smo.append((cx, orix))

        #Encoder-Decoder Link
        if self.sc:
            ex, orix = self.conver(ex)
        else:
            ex, orix = ex, ex
        smo.append((ex, orix))
        
        #Decoder
        dx, orix, cx = self.decs[0](x[:,-self.dec_seq_len:], enc=ex)
        smo.append((cx,orix))
        for dec in self.decs[1:]:
            dx, orix, cx = dec(dx,enc=ex)
            smo.append((cx,orix))
            
        #generative output
        if self.convout:
            x = self.out_fc1(dx)
        else:
            batch_size = x.size(0)
            x = self.out_fc1(dx.flatten(start_dim=1))
            x = x.view(batch_size, -1, self.output_size)
        return x, smo