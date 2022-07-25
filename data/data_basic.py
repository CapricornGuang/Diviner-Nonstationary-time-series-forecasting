import os
import numpy as np
import pandas as pd

import torch
from sklearn.preprocessing import StandardScaler


import warnings
warnings.filterwarnings('ignore')

class DatasetBasic(torch.utils.data.Dataset):
    def __init__(self, args, mode):
        #variables for mode selection
        assert mode in ['train', 'valid', 'test'], 'KeyError:{} not in options, options:[train, valid, test]'.format(mode)
        type_map = {'train':0, 'valid':1, 'test':2}
        self.mode_index = type_map[mode]
        self.model = args.model
        #variables for collecting data
        self.root_data_path = os.path.join(args.root_path, args.data_path)
        self.features = args.features
        self.target = args.target
        self.pattern_length = args.pattern_length
        #variables for preprocessing data
        self.out_scale = args.out_scale
        self.enc_seq_len = args.enc_seq_len
        self.out_seq_len = args.out_seq_len
        self.input_scaler = StandardScaler()
        self.label_scaler = StandardScaler()

    def __read_data__(self):
        raise NotImplementedError
    
    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def fit_input_scaler(self, data):
        self.input_scaler.fit(data.values)
        return None
    
    def fit_label_scaler(self, data):
        self.label_scaler.fit(data.values)
        return None

    def input_transform(self, data):
        return self.input_scaler.transform(data)

    def label_transform(self, data):
        return self.label_scaler.transform(data)

    def inverse_input_transform(self, data):
        return self.input_scaler.inverse_transform(data)

    def inverse_label_transform(self, data):
        return self.label_scaler.inverse_transform(data)