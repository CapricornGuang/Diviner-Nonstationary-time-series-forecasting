import copy
import pandas as pd
import numpy as np
import torch
import numpy as np
from torch import optim

'''Basic Operation'''
def prioritize_ele(lst, ele):
    lst_tmp = copy.deepcopy(lst)
    try:
        lst_tmp.remove(ele)
    except ValueError:
        pass
    lst_tmp.insert(0,ele)
    return lst_tmp

def get_average(lst):
    return sum(lst)/len(lst)


def list_mul(lst, M, B=0):
    '''Multiplate lst with f'''
    assert type(lst) is list, TypeError('{} is not a list'.format(lst))
    type_parser = {
        (list, list): lambda x,y:[int(int(val*m)+b) for val,m,b in zip(lst,x,y)],
        (list, int): lambda x,y:[int(int(val*m)+y) for val,m in zip(lst,x)],
        (int, list): lambda x,y:[int(int(val*x)+b) for val, b in zip(lst,y)],
        (int, int): lambda x,y: [int(int(val*x)+y) for val in lst]
    }
    return type_parser[(type(M), type(B))](M,B)




'''Data Preprocessing'''
def get_time_series_matrix(ts, pattern_length):
    '''
    ts: time series, nparray
    '''
    assert len(ts.shape)==2
    if ts.shape[1]==1:
        return ts.flatten().reshape(-1, pattern_length)   #seq_len, pattern_length 
    else:
        channel_num = ts.shape[1]
        return ts.T.reshape(channel_num, -1, pattern_length).transpose(1,0,2) #seq_len, channel_num, pattern_length

def interpolate_nan(df):
    '''
    this function is for padding the missing data
    we first use linear interpolation to process the missing data not at the both ends.
    then, we use forwar interpolation and backward interpolation to process the missing
    data at left and right ends respectively
    '''
    df.interpolate(method='linear',inplace=True)
    df.interpolate(method='ffill', limit_direction='forward',inplace=True)
    df.interpolate(method='bfill', limit_direction='backward',inplace=True)
    return df