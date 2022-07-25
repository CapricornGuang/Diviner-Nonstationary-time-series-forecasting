import torch
import warnings, argparse, json, ast, sys, os, re
from utils import tools
warnings.filterwarnings('ignore')
from exp import exp_diviner

parser = argparse.ArgumentParser(description='Diviner: long-term forecasting')
#model selection
parser.add_argument('--model', type=str, required=True, default='diviner', 
    help='''model of experiment, options: [diviner, diviner-cg, diviner-sc, diviner-diff, diviner-self masked,vanilla transformer],
    diviner: the standard diviner model
    diviner-cg: diviner without convolutional generator
    diviner-sc: diviner without smoothing attention mechanism
    diviner-diff: diviner without difference attention module.
    diviner-masked: diviner without self-masked structure''')
parser.add_argument('--ddblock', type=bool, default=True, 
    help='''option for employing difference attention module, options:[True, False], 
    when ddblock=False, a vanilla attention is employed.''')
parser.add_argument('--sc', type=bool, default=True, help='option for employing smoothing attention mechanism')
parser.add_argument('--self_masked', type=bool, default=True, help='option for self-masked structure')

#data selection
parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--predict_length', type=int, required=True, help='predict length')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--split_ratio1', type=ast.literal_eval, default="[0, 0.7, 0.8]")
parser.add_argument('--split_ratio2', type=ast.literal_eval, default="[0.7, 0.8, 1.0]")
parser.add_argument('--features', nargs='+', type=str, default=[], help='features used for training' )
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--pattern_length', type=int, default=24, 
help='option related to sample freq, representing the total time steps in a temporal unit')
parser.add_argument('--out_scale', action='store_true', help='option to excute data standardization when data comes into models')
parser.add_argument('--enc_seq_len', type=int, default=60, required=True, help='input pattern num of Diviner encoder')
parser.add_argument('--out_seq_len', type=int, default=30, required=True, help='output pattern num of Diviner generator')

#model-parameter selection
parser.add_argument('--dec_seq_len', type=int, default=30, required=True, help='input pattern num of Diviner decoder')
parser.add_argument('--dim_input', type=int, default=24, help='dimension of input data')
parser.add_argument('--dim_output', type=int, default=24, help='dimension of output data')

parser.add_argument('--dim_val', type=int, default=24, help='dimension of the embedded data')
parser.add_argument('--dim_attn', type=int, default=24, help='dimension of the self-attention in smoothing filter attention mechnism and difference attention module')
parser.add_argument('--dim_attn_channel', type=int, default=24, help='dimension of the channel-attention')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads of the self-attention in smoothing filter attention mechnism and difference attention module')
parser.add_argument('--n_encoder_layers', type=int, default=3, help='num of encoder layers')
parser.add_argument('--n_decoder_layers', type=int, default=2, help='num of decoder layers')
parser.add_argument('--conv_out', type=ast.literal_eval, default="{'use':True, 'kernel':5, 'layers':3}", help='option for conventional generator')

#train-parameter selection
parser.add_argumemt('--test', action='store_true', help='option for directly testing models')

parser.add_argument('--check_points', type=str, default='./checkpoints', help='The root of trained model address')
parser.add_argument('--load_check_points', type=str, default='', help='option for loading checkpoints model to train or test')
parser.add_argument('--verbose', action='store_true', help='option for displaying the saving-information')
parser.add_argument('--early_stop', action='store_true', help='option for employing early_stop')
parser.add_argument('--patience', type=int, default=7, help='option for the early stop algorithm')
parser.add_argument('--delta', type=float, default=0., help='option controling the bias when early stop is employed')

parser.add_argument('--batch_size', type=int, default=64, help='batch size of train input data')
parser.add_argument('--train_epochs', type=int, default=5, help='option for shuffling train input data')
parser.add_argument('--shuffle', action='store_true', help='option for shuffling train input data')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--drop_last', action='store_true', help='option for droping last batch')

parser.add_argument('--optimizer', type=str, default='AdamW', help='option for selecting optimizer, options:[Adam, AdamW, SGD, RMSprop,Adagrad, Adadelta]')
parser.add_argument('--lr', type=float, default=1e-3, help='option for learning rate')
parser.add_argument('--loss', type=str, default='mse', help='option for loss function, option:[mse, mae]')
parser.add_argument('--smo_loss', action='store_true', help='option for constraint smoothing filter attention mechanism')
parser.add_argument('--dynamic', action='store_true', help='option for ignoring larger loss in loss calculation')
parser.add_argument('--dynamic_ratio', type=float, default=0.8, help='option for setting a ratio to filter out the larger data in a series sorted with an ascending order.')

parser.add_argument('--use_gpu', action='store_true', help='option for training device')
parser.add_argument('--gpu_id', type=int, default=0, help='option for employed gpu index')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
parser.add_argument('--device_ids', type=list, default=0, help='devices used for data parallel')

args = parser.parse_args()
data_parser = {
    'ETTh1':{'root_path':'./data/ETT/', 'data_path':'ETTh1.csv','target':'OT', 'pattern_length':24, },
    'ETTh2':{'root_path':'./data/ETT/','data_path':'ETTh2.csv','target':'OT', 'pattern_length':24},
    'ETTm1':{'root_path':'./data/ETT/','data_path':'ETTm1.csv','target':'OT', 'pattern_length':24*4},
    'WTH':{'root_path':'./data/WTH/','data_path':'WTH.csv','target':'WetBulbFarenheit', 'pattern_length':24*6},
    'ECL':{'root_path':'./data/ECL/','data_path':'ECL.csv','target':'MT_320', 'pattern_length':24},
    'Exchange':{'root_path':'./data/Exchange/','data_path':'gold.csv','target':'price', 'pattern_length':1},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data_path']
    args.target = data_info['target']
    args.pattern_length = data_info['pattern_length']
    args.root_path = data_info['root_path']
    args.dim_output = args.pattern_length
    args.dim_input = args.pattern_length
    
model_parser = {
    'diviner':{'conv_out':True, 'sc':True, 'ddblock':True, 'self_masked':True},
    'diviner-cg':{'conv_out':False, 'sc':True, 'ddblock':True, 'self_masked':True},
    'diviner-sc':{'conv_out':True, 'sc':False, 'ddblock':True, 'self_masked':True},
    'diviner-df':{'conv_out':True, 'sc':True, 'ddblock':False, 'self_masked':True},
    'diviner-sm':{'conv_out':True, 'sc':True, 'ddblock':True, 'self_masked':False},
    'vanilla-transformer':{'conv_out':False, 'sc':False, 'ddblock':False, 'self_masked':False},
}

if args.model in model_parser.keys():
    data_info = model_parser[args.model]
    args.conv_out['use'] = data_info['conv_out']
    args.sc = data_info['sc']
    args.ddblock = data_info['ddblock']
    args.self_masked = data_info['self_masked']
elif args.model == 'customized':
    pass
else:
    raise KeyError('{} not in options'.format(args.model))
if args.data == 'WTH' and args.predict_length==2016:
    args.split_ratio1 = [0, 0.65, 0.8]
    args.split_ratio2 = [0.65, 0.8, 1.0]
if args.data == 'WTH' and args.predict_length==4032:
    args.split_ratio1 = [0, 0.50, 0.75]
    args.split_ratio2 = [0.50, 0.75, 1.0]

if args.test:
    assert len(args.load_check_points)!= 0

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu_id = args.device_ids[0]


args.features = tools.prioritize_ele(args.features, args.target)
args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

print('Args in experiment:')
print(args)


settings = 'dimVal{}_dimInput{}_encSeqLen{}_decSeqLen{}_outSeqLen{}_encoderLayers{}_decoderLayers{}_nheads{}_sc{}_ddblock{}_selfMasked{}'.format(
    args.dim_val, args.dim_input,
    args.enc_seq_len, args.dec_seq_len, args.out_seq_len, 
    args.n_encoder_layers, args.n_decoder_layers,
    args.n_heads, 
    args.sc, args.ddblock, args.self_masked)

root_path = os.path.join(args.check_points, args.data, str(args.predict_length))
if not os.path.exists(root_path):
    os.makedirs(root_path)
bash_path = os.path.join(root_path, 'run.sh')
f = open(bash_path, 'w', encoding='utf-8')
sys_bash = 'python -u main.py '+' '.join(sys.argv[1:])
regex = re.compile('--load_check_point=.*? ')
sys_bash = re.sub(regex,'',sys_bash)
regex = re.compile('--lr=.*')
sys_bash = re.sub(regex,'',sys_bash)
f.write(sys_bash)
f.close()



exp_env = exp_diviner.Exp_Diviner(args)

print('>>>>>>>start training >>>>>>>>>>>>>>>>>>>>>>>>>>')
if not args.test:
    exp_env.train(settings)

print('>>>>>>>testing<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
MSE, MAE = exp_env.test(settings)
torch.cuda.empty_cache()
