import torch
import os

class Exp_Basic():
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_network().to(self.device)

    def _build_network(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu_id) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu_id))
            print('Use GPU: cuda:{}'.format(self.args.gpu_id))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self):
        raise NotImplementedError

    def valid(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

