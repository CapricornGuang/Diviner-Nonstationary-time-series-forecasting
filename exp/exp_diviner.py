from importlib.resources import path
import torch
from torch.utils.data import DataLoader
import os, time
import numpy as np
from exp.exp_basic import Exp_Basic
from exp import utils as exp_utils

from models.networks import Diviner
from data.data_loader import DatasetETT, DatasetCustom

from utils import tools
from matplotlib import pyplot as plt

class Exp_Diviner(Exp_Basic):
    def __init__(self, args):
        super(Exp_Diviner, self).__init__(args)
        
    def _build_network(self):
        assert self.args.model in ['diviner', 'diviner-cg', 'diviner-sc', 'diviner-diff', 
        'diviner-self masked', 'vanilla transformer'], 'Exp_Diviner not support {}'.format(self.args.model)
        print('start building network...')
        net = Diviner(
            self.args.dim_val, 
            self.args.dim_attn, 
            self.args.dim_attn_channel,
            self.args.dim_input,
            self.args.dim_output, 
            self.args.enc_seq_len, 
            self.args.dec_seq_len, 
            self.args.out_seq_len, 
            self.args.n_decoder_layers, 
            self.args.n_encoder_layers, 
            self.args.n_heads, 
            self.args.conv_out, 
            self.args.sc, 
            self.args.ddblock, 
            self.args.self_masked)
        print('''network parameters:
        dim_val:{},dim_attn:{},dim_input:{},dim_output:{},
        enc_seq_len:{}, dec_seq_len:{}, out_seq_len:{},
        n_decoder_layers:{}, n_encoder_layers:{},
        n_heads:{}, conv_out:{}
        sc:{}, ddblock:{}, self_masked:{}
        '''.format(self.args.dim_val,self.args.dim_attn,
        self.args.dim_input,self.args.dim_output, 
        self.args.enc_seq_len, self.args.dec_seq_len, self.args.out_seq_len, 
        self.args.n_decoder_layers, self.args.n_encoder_layers, 
        self.args.n_heads, self.args.conv_out, 
        self.args.sc, self.args.ddblock, self.args.self_masked))
        
        if len(self.args.load_check_points) != 0:
            net.load_state_dict(torch.load(self.args.load_check_points))

        if self.args.use_multi_gpu and self.args.use_gpu:
            net = exp_utils.to_data_parallel(net, self.args.device_ids)
        return net
    
    def _get_data(self, mode):
        args = self.args
        data_dict = {
            'ETTh1':DatasetETT,
            'ETTh2':DatasetETT,
            'ETTm1':DatasetETT,
            'ETTm2':DatasetETT,
            'WTH':DatasetCustom,
            'ECL':DatasetCustom,
            'Exchange':DatasetCustom,
        }
        mode_parser = {
            'train': {'shuffle':args.shuffle, 'drop_last': args.drop_last, 'num_workers':args.drop_last},
            'valid': {'shuffle':True, 'drop_last':False, 'num_workers':0},
            'test': {'shuffle':False, 'drop_last':False, 'num_workers':0}
        }
        mode_info = mode_parser[mode]
        data_set = data_dict[args.data](args, mode)
        print('dataset size of {}:{}'.format(mode, len(data_set)))

        data_loader = DataLoader(data_set, batch_size=args.batch_size, 
        shuffle=mode_info['shuffle'],
        num_workers=mode_info['num_workers'],
        drop_last = mode_info['drop_last'])
        return data_set, data_loader

    def valid(self, vali_loader):
        self.model.eval()
        print('validing...')
        valid_loss_records = []
        for X, y in vali_loader:
            outputs = self._process_one_batch(X, y, loss_flag=self.args.loss, dynamic_loss_flag=self.args.dynamic)
            valid_loss_records.append(outputs['loss'])
        valid_loss = tools.get_average(valid_loss_records)
        self.model.train()
        return valid_loss

    def test(self, settings):
        self.model.eval()
        test_dataset, test_loader = self._get_data(mode = 'test')
        print('testing...')
        MSE_records, MAE_records = [], []
        predict_values = []
        original_values = []
        for X, y in test_loader:
            with torch.no_grad():
                mse_res = self._process_one_batch(X, y, loss_flag='mse', dynamic_loss_flag=False, flatten=True, predict_length=self.args.predict_length)
                mae_res = self._process_one_batch(X, y, loss_flag='mae', dynamic_loss_flag=False, flatten=True, predict_length=self.args.predict_length)
            mse_res['net_out'] = mse_res['net_out'].detach().cpu().numpy()
            mae_res['net_out'] = mae_res['net_out'].detach().cpu().numpy()
            if not self.args.out_scale:
                mse_res['net_out'] = test_dataset.inverse_label_transform(mse_res['net_out'])
                mae_res['net_out'] = test_dataset.inverse_label_transform(mae_res['net_out'])
            MSE_records.append(mse_res['loss']); MAE_records.append(mae_res['loss'])
            predict_values.append(out['net_out'])
            original_values.append(out['label'].detach().cpu().numpy())
        
        MSE = tools.get_average(MSE_records)
        MAE = tools.get_average(MAE_records)
        np.save('predict_values.npy', predict_values)
        np.save('original_values.npy', original_values)
        self.model.train()
        print('{}-{} dataset experimental results'.format(self.args.data, self.args.predict_length))
        print('MSE:{}, MAE:{}'.format(MSE, MAE))
        return MSE, MAE


    def train(self, settings):
        print('start collecting data...')
        _, train_loader = self._get_data(mode = 'train')
        _, valid_loader = self._get_data(mode = 'valid')
        print('collecting data done...')
        early_stopping = exp_utils.EarlyStopping(self.args.patience, self.args.verbose and self.args.early_stop, self.args.delta)
        optimizer = exp_utils.get_optimizer(self.args.optimizer, self.model.parameters(), self.args.lr)
        
        root_path = os.path.join(self.args.check_points, self.args.data, str(self.args.predict_length))
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        model_path = os.path.join(root_path, 'checkpoints_'+settings+'.ckpt')

        print(f'start training')
        time_now = time.time(); train_steps = len(train_loader)
        for epoch in range(self.args.train_epochs):
            iter_count = 0; train_loss_record = []
            epoch_time = time.time()
            for i, (X, y) in enumerate(train_loader):
                iter_count += 1
                outputs = self._process_one_batch(X, y, loss_flag=self.args.loss, dynamic_loss_flag=self.args.dynamic)
                f_loss = outputs['loss']
                s_loss = tools.get_average([((item1-item2)**2).mean() for item1, item2 in outputs['model_inter']])
                loss = s_loss+f_loss if self.args.smo_loss else f_loss
                train_loss_record.append(f_loss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                "Timing functionality is refered from https://github.com/zhouhaoyi/Informer2020"
                if (iter_count+1) % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(iter_count + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
            train_loss = tools.get_average(train_loss_record)
            valid_loss = self.valid(valid_loader)
            print("Epoch{}: cost time={} | train loss={}, valid loss={}".format(epoch+1, 
            time.time()-epoch_time, train_loss, valid_loss))
            if self.args.early_stop:
                valid_loss_pri = early_stopping.val_loss_min
                early_stopping(valid_loss)
                valid_loss_aft = early_stopping.val_loss_min
                if early_stopping.save:
                    self._establish_check_points(valid_loss_pri, valid_loss_aft, model_path, self.args.verbose)
                if early_stopping.early_stop:
                    print("degredation patience is out, triggle early-stopping")
                    break
        if self.args.early_stop:
            self.model.load_state_dict(torch.load(model_path))
        else:
            self._establish_check_points(None, None, model_path, False)
        print(f'The model is saving at %s'%(model_path))
        return self.model

    def _process_one_batch(self, batch_x, batch_y=None, flatten=False, loss_flag=None, dynamic_loss_flag=None, predict_length=None):
        '''
        batch_x: model input
        batch_y: expected output
        flatten: an option for turning model output into a series
        inter: an option for getting internal results when model inffering. 
        
        '''
        #batch_x: [batch_size, seq_len, channel_num?, dim_val]
        batch_x = batch_x.to(self.device)
        net_out, smo = self.model(batch_x)
        # print('input_nan',torch.isnan(batch_x).sum().item(),'output_nan:',torch.isnan(net_out).sum().item())

        outputs = {'label':None, 'net_out':None, 'loss':None, 'model_inter':smo}
        label = batch_y
        if batch_y is not None:
            batch_y = batch_y.to(self.device)
            outputs['loss']=self._get_criterion_loss(net_out, batch_y, loss_flag, dynamic_loss_flag, predict_length)
        if flatten == True:
            net_out = net_out.flatten(start_dim=1).flatten()
            if batch_y is not None:
                label = label.flatten(start_dim=1).flatten()
        outputs['net_out'] = net_out
        outputs['label'] = label
        return outputs

    def _get_criterion_loss(self, output, label, types, dynamic, predict_length):
        val_loss = exp_utils.loss_criterion(output, label, types)
        return exp_utils.dynamic_criterion(val_loss, dynamic, self.args.dynamic_ratio, self.device, predict_length)

    def _establish_check_points(self, val_loss_pri, val_loss_aft, model_path, verbose):
        if verbose:
            print(f'Validation loss decreased ({val_loss_pri:.6f} --> {val_loss_aft:.6f}).  Saving model ...')
        torch.save(self.model.state_dict(), model_path)
