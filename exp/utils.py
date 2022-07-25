import torch
import numpy as np

'''data parallel'''
def to_data_parallel(net, devices):
    return torch.nn.DataParallel(net, devices)

'''loss'''
def get_dynamic_weight_loss(X, device,ratio=0.8):
    '''
    this function is for eliminating the influence of abnormal points
    we will sort the loss value in X in natural order
    for the value whose order exceeding an appointed ratio we will delete it or multiple a scalar<1
    X: [batch_size, ]
    ratio: the value whose order exceeding ratio*len(X) will get punishment
    '''
    length = len(X)
    X = torch.sort(X)[0]
    pivot = int(length*ratio)
    weight = torch.ones(length).to(device)
    weight[pivot:] = 0
    return (X*weight).sum()/(weight.sum())


def loss_criterion(output, label, types):
    assert types in ['mae', 'mse']
    if types == 'mae':
        return torch.abs(output-label)
    else:
        return torch.square(output-label)



def dynamic_criterion(val_loss, dynamic, ratio, device, predict_length):
    val_loss_flatten = val_loss.flatten(start_dim=1)

    if predict_length is None:
        return get_dynamic_weight_loss(val_loss_flatten.mean(1),
            device, ratio) if dynamic else val_loss_flatten.mean()
    else:
        assert dynamic == False
        return val_loss_flatten[:, :predict_length].mean()

'''Optimizer'''
def get_optimizer(types, net_param, lr):
    options = ['Adam', 'AdamW', 'SGD', 'RMSprop','Adagrad', 'Adadelta']
    assert types in options, '{} is not supported, options:{}'.format(types, options)
    return eval('torch.optim.{optimizer}'.format(optimizer=types))([{'params': net_param}], lr=lr)


'''Training Strategy'''
"The code here is refered from https://github.com/zhouhaoyi/Informer2020"
def adjust_learning_rate(optimizer, epoch, args, decay_rate=0.8):
    if args.lradj=='type1':
        lr_adjust = {epoch: args.lr * (decay_rate ** ((epoch-1) // 1))}
    elif args.lradj=='type2':
        lr_adjust = {
            2: 1e-3, 4: 5e-4, 6: 1e-4, 8: 1e-4, 
            10: 5e-5, 15: 1e-5, 20: 1e-5
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))



"The code here is refered from https://github.com/zhouhaoyi/Informer2020"
class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save = False
    def __call__(self, val_loss):
        score = -val_loss; self.save = False

        if self.best_score is None:
            self.best_score = score
            self.save = True
            self.val_loss_min = val_loss

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose == True:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            self.save = True
            self.val_loss_min = val_loss