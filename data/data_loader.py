import os
import torch
import numpy as np
import pandas as pd
import utils.tools as tools
from data.data_basic import DatasetBasic

TRAIN_INDEX = 0
class DatasetETT(DatasetBasic):
    def __init__(self, args, mode='train'):
        super(DatasetETT, self).__init__(args, mode)

        self.df_raw = None
        self.__read_data__()

        split_stamp1 = [0, 12*30-self.enc_seq_len, 12*30+4*30-self.enc_seq_len]
        split_stamp2 = [12*30, 12*30+4*30, 12*30+8*30]
        self.border1s = tools.list_mul(split_stamp1, self.pattern_length)
        self.border2s = tools.list_mul(split_stamp2, self.pattern_length)

        #! To benchmark other methods, self.__split_scale_data__() should not be changed,
        #! To customize other methods, self.__getitem__() should be configured
        self.data_input, self.data_label = None, None
        self.__split_scale_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(self.root_data_path)
        self.total_pattern_num = len(df_raw)//self.pattern_length
        df_raw = df_raw[:self.total_pattern_num*self.pattern_length]
        self.df_raw = tools.interpolate_nan(df_raw)
        self.len_raw = len(self.df_raw)

    def __split_scale_data__(self):
        '''
        This function is to slice and preprocess raw data into our model-input and model-label
        data_input: [pattern_length, pattern_num]
        data_output: [pattern_length, pattern_num]
        '''
        #! 'raw' means the whole data of the corresponding mode
        df_raw = self.df_raw
        border1, border2= self.border1s[self.mode_index], self.border2s[self.mode_index]
        border1_train, border2_train = self.border1s[0], self.border2s[0]

        #Get data_input
        df_raw_input, df_raw_label = df_raw[self.features], df_raw[[self.target]]
        data_input, data_label = df_raw_input[border1:border2], df_raw_label[border1:border2]
        self.fit_input_scaler(df_raw_input[border1_train:border2_train])
        data_input = self.input_transform(data_input.values)

        #Get data_label
        self.fit_label_scaler(df_raw_label[border1_train:border2_train])
        data_label = self.label_transform(data_label.values)
        
        self.data_input = data_input
        self.data_label = data_label

    def __getitem__(self, index):
        #data_input: [pattern_num, pattern_length]
        #data_label: [pattern_num, pattern_length]
        if self.model in ['diviner', 'diviner-cg', 'diviner-sc', 'diviner-diff', 'diviner-self masked']:
            data_input_matrix = tools.get_time_series_matrix(self.data_input, self.pattern_length)
            data_label_matrix = tools.get_time_series_matrix(self.data_label, self.pattern_length)

            #Get model_input
            x_begin = index
            x_end = x_begin+self.enc_seq_len

            data_x = data_input_matrix[x_begin:x_end]
            data_x = torch.from_numpy(data_x.astype('float32'))

            #Get model_label
            y_begin = x_end
            y_end = y_begin+self.out_seq_len
            data_y = data_label_matrix[y_begin:y_end]
            data_y = torch.from_numpy(data_y.astype('float32'))
            return data_x, data_y
        else:
            raise KeyError('{} is not supported'.format(self.model))

    def __len__(self):
        #data_input: [pattern_num, pattern_length]
        data_input_matrix = tools.get_time_series_matrix(self.data_input, self.pattern_length)
        pattern_num = data_input_matrix.shape[0]
        return pattern_num-self.enc_seq_len-self.out_seq_len

class DatasetCustom(DatasetBasic):
    def __init__(self, args, mode='train'):
        super(DatasetCustom, self).__init__(args, mode)

        self.df_raw = None
        self.__read_data__()

        #* Data should be splited along the dimension of days/weeks
        split_ratio1 = args.split_ratio1
        split_ratio2 = args.split_ratio2
        split_stamp1 = self._transform_ratio_to_stamp(split_ratio1)
        split_stamp2 = self._transform_ratio_to_stamp(split_ratio2)
        self.border1s = tools.list_mul(split_stamp1, self.pattern_length)
        self.border2s = tools.list_mul(split_stamp2, self.pattern_length)

        #! To benchmark other methods, self.__read_data__() should not be changed,
        #! To customize other methods, self.__getitem__() should be configured
        self.data_input, self.data_label = None, None
        self.__split_scale_data__()


    def __read_data__(self):
        df_raw = pd.read_csv(self.root_data_path)
        self.total_pattern_num = len(df_raw)//self.pattern_length
        df_raw = df_raw[:self.total_pattern_num*self.pattern_length]
        self.df_raw = tools.interpolate_nan(df_raw)
        self.len_raw = len(self.df_raw)

    def __split_scale_data__(self):
        '''
        This function is to slice and preprocess raw data into our model-input and model-label
        data_input: [pattern_length, pattern_num]
        data_output: [pattern_length, pattern_num]
        '''
        #* 'raw' means the whole data of the corresponding mode
        df_raw = self.df_raw
        df_raw_input, df_raw_label = df_raw[self.features], df_raw[[self.target]]
        border1, border2= self.border1s[self.mode_index], self.border2s[self.mode_index]
        border1_train, border2_train = self.border1s[0], self.border2s[0]

        #Get data_input
        data_input, data_label = df_raw_input[border1:border2], df_raw_label[border1:border2]
        self.fit_input_scaler(df_raw_input[border1_train:border2_train])
        data_input = self.input_transform(data_input.values)

        #Get data_label

        self.fit_label_scaler(df_raw_label[border1_train:border2_train])
        data_label = self.label_transform(data_label.values)

        self.data_input = data_input
        self.data_label = data_label

    def __getitem__(self, index):
        #data_input: [pattern_num, pattern_length]
        #data_label: [pattern_num, pattern_length]
        if self.model in ['diviner', 'diviner-cg', 'diviner-sc', 'diviner-diff', 'diviner-self masked']:
            #Get model_input
            x_begin = index
            x_end = x_begin+self.enc_seq_len*self.pattern_length
            data_x = self.data_input[x_begin:x_end]
            data_x = tools.get_time_series_matrix(data_x, self.pattern_length)
            data_x = torch.from_numpy(data_x.astype('float32'))

            #Get model_label
            y_begin = x_end
            y_end = y_begin+self.out_seq_len*self.pattern_length
            data_y = self.data_label[y_begin:y_end]
            data_y = tools.get_time_series_matrix(data_y, self.pattern_length)
            data_y = torch.from_numpy(data_y.astype('float32'))
            return data_x, data_y
        else:
            raise KeyError('{} is not supported'.format(self.model))

    def __len__(self):
        #data_input: [pattern_num, channels(?), pattern_length]
        data_input_matrix = tools.get_time_series_matrix(self.data_input, self.pattern_length)
        total_length = data_input_matrix.shape[0]*self.pattern_length
        enc_length = self.enc_seq_len*self.pattern_length
        out_length = self.out_seq_len*self.pattern_length
        return total_length-enc_length-out_length

    def _transform_ratio_to_stamp(self, split_ratio):
        split_stamp = tools.list_mul(split_ratio, self.total_pattern_num, [0, -self.enc_seq_len, -self.enc_seq_len])
        return split_stamp