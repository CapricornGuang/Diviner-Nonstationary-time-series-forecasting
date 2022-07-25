# Diviner: Long-term time series forecasting for 5G cellular network capacity planning
![Python 3.6.5](https://img.shields.io/badge/python-3.6.5-green.svg?style=plastic)
![PyTorch 1.2](https://img.shields.io/badge/PyTorch-1.12.0-orange?style=plastic)
![License CC BY-NC-SA](https://img.shields.io/badge/license-CC_BY--NC--SA--green.svg?style=plastic)

This is the origin Pytorch implementation of Diviner in the following paper: 
[Diviner: Long-term time series forecasting for 5G cellular network capacity planning](???).




## Diviner Framework
Diviner aims at exploring the multi-scale stable regularities within time-series data, whose data distribution varies over time. To this end, we propose `Smoothing Filter Attention Mechanism` to filter out random components and adjust the feature scale layer-by-layer. Simultaneously, a `Difference Attention Module` is designed to calculate long- and short- range dependencies by capturing the stable shifts at the corresponding scale. The recipe of our work is to mine constants in change!

<p align="center">
<img src=".\.img/Framework.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 1.</b> The illustration of Diviner framework.
</p>

## Data
All the data you need have been 
can be found in the `data` folder. The following figure illustrates a demo of the NPT data. Note that the input of each dataset is standarlized.
<p align="center">
<img src=".\.img/Figure1.png" height = "256" alt="" align=center />
<br><br>
<b>Figure 2.</b> An example of the NPT data.
</p>

## Requirements

- Python 3.6+
- numpy == 1.21.6
- pandas == 1.3.5
- scikit_learn == 1.0.2
- torch == 1.12.0+cu113

Dependencies can be installed using the following command:
```bash
pip install -r requirements.txt
```


## Reproducibility
You can follow our work by easily cloning our repository. To ease the reproducibility of results, the checkpoints of each dataset is provided in `.\checkpoints` folder. We note that you can use the **run.sh** file in each sub-folders of `.\checkpoints` to test our pretrained model.

```bash
git clone https://github.com/CapricornGuang/NetworkForecast.git
# the content of <> depends on which pretrained model you want to use.
# you can check the `checkpoints` folder to get options.
bash NetworkForecast/checkpoints/<dataset>/<predict span>/run.sh
```
The tree below illustrates the organization of this project.
```bash
├── NetworkForecast
│   ├── checkpoints # folder to store pretrained models
│   ├── data
│   │   ├──ECL、ETT、Exchange、WTH
│   │   ├──data_basic.py  #Father class of all dataloader subclass
│   │   ├──data_loader.py #Split and Organize raw data into the (batch, seq, dim) formats
│   ├── model 
│   │   ├──attn.py #Vanilla Attention Implement
│   │   ├──blocks.py #Proposed Mechanism Implement
│   │   ├──embed.py #Input layer Implement
│   │   ├──layers.py #Encoder, Decoder, OneStep-Generator Implement
│   │   ├──networks.py #Network Structure
│   ├── scripts 
```

## Usage
Here we present a simple demo of training your Diviner model.
```bash
#Traing <Diviner model> on ETTh1 dataset from scratch.
python -u main.py --model=diviner --data=ETTh1 --predict_length=336 --enc_seq_len=30 --out_seq_len=14 --dec_seq_len=14 --dim_val=24 --dim_attn=12 --dim_attn_channel=48 --n_heads=6 --n_encoder_layers=3 --n_decoder_layers=2 --batch_size=32 --train_epochs=100 --use_gpu --smo_loss --dynamic --early_stop --shuffle --verbose --out_scale

#Testing <Diviner model> on ETTh1 dataset from scratch.
python -u main.py --model=diviner --data=ETTh1 --predict_length=336 --enc_seq_len=30 --out_seq_len=14 --dec_seq_len=14 --dim_val=24 --dim_attn=12 --dim_attn_channel=48 --n_heads=6 --n_encoder_layers=3 --n_decoder_layers=2 --batch_size=32 --train_epochs=100 --use_gpu --smo_loss --dynamic --early_stop --shuffle --verbose --out_scale --test 
--load_check_points=.\checkpoints\ETTh1\336\diviner_checkpoints.ckpt 
```


More parameter information please refer to `main.py`. The detailed descriptions about the arguments are as following (the <font color="blue">blue</font> color arguments should be carefully considered, the <font color="green">green</font> color arguments is optional by default).
<table>
    <tr>
        <td><b>Argument type</b></td> 
        <td><b>Argument name</b></td> 
        <td><b>Description of argument</b></td> 
   </tr>
    <tr>
        <td rowspan="2">Model Selection</td>    
  		 <td><font color="blue">model</font></td> 
      	 <td>Options:[diviner, diviner-cg, diviner-sc,diviner-diff, diviner-self masked]</br>
          diviner: the standard diviner mode.l</br>
          diviner-cg: diviner without convolutional generator.</br>
          diviner-sc: diviner without smoothing attention mechanism</br>
          diviner-diff: diviner without difference attention module.</br>
          diviner-masked: diviner without self-masked structure.</br>
         </td> 
    </tr>
    <tr>
        <td>sc, ddblock, sc, self_masked</td> 
        <td>Options for ablation study. These parameters are configured automatically when the `model` is assigned.</td>    
    </tr>
    <tr>
        <td rowspan="7">Data arguments</td>    
  		 <td><font color="blue">data</font></td> 
      	 <td>The dataset name</td> 
    </tr>
    <tr>
        <td>root_path</td> 
        <td>The root path of the data file, default: './data/ETT'</td>    
    </tr>
    <tr>
        <td>data_path</td> 
        <td>The data file name, default: 'ETTh1'</td>    
    </tr>
    <tr>
        <td><font color="green">features</font></td> 
        <td>The features employed for data input</td>    
    </tr>
    <tr>
        <td>target</td> 
        <td>The target features to predict</td>    
    </tr>
    <tr>
        <td><font color="blue">out_scale</font></td> 
        <td>Option for employing standarlization for output of the model</td>    
    </tr>
    <tr>
        <td>pattern_length</td> 
        <td>Total time steps in a temporal unit</td>    
    </tr>
    <tr>
        <td rowspan="12">Model arguments</td>    
  		 <td><font color="blue">enc_seq_len</font></td> 
      	 <td>The input pattern num of Diviner encoder</td> 
    </tr>
    <tr>
        <td><font color="blue">dec_seq_len</font></td> 
        <td>The input pattern num of Diviner decoder</td>    
    </tr>
    <tr>
        <td><font color="blue">out_seq_len</font></td> 
        <td>output pattern num of Diviner generator</td>    
    </tr>
    <tr>
        <td>dim_input</td> 
        <td>The dimension of input data</td>    
    </tr>
    <tr>
        <td>dim_output</td> 
        <td>The dimension of output data</td>    
    </tr>
    <tr>
        <td><font color="blue">dim_val</font></td> 
        <td>The dimension of the embedded data</td>    
    </tr>
    <tr>
        <td><font color="blue">dim_attn</font></td> 
        <td>The dimension of Q,K,V in self-attention of Smoothing Filter Attention Mechanism and Difference Attention Module</td>    
    </tr>
    <tr>
        <td><font color="green">dim_attn_channel</font></td> 
        <td>The dimension of Q,K,V in self-attention of Multi-Channel Embedded. <i>This argument ought  to be set when you employ multi-features</i> </td>
    </tr>
    <tr>
        <td><font color="blue">n_heads</font></td> 
        <td>The number of heads in self-attention mechanism</td>    
    </tr>
     <tr>
        <td><font color="blue">n_encoder_layers</font></td> 
        <td>The number of encoder layers</td> 
    </tr>
    <tr>
        <td><font color="blue">n_decoder_layers</font></td> 
        <td>The number of decoder layers</td> 
    </tr>
    <tr>
        <td><font color="green">conv_out</font></td> 
        <td>default:{'use':True, 'kernel':5, 'layers':3}<br/>
        use=True: Use convolutional one-step generator<br/>
           use=False: Use linear one-step generator
        </td> 
    </tr>
    <tr>
        <td rowspan="7">Train arguments</td>    
  		 <td><font color="green">test</font></td> 
      	 <td>Option for directly testing the appointed models (by assigned load_check_points argument)</td>
    </tr>
    <tr>
        <td><font color="green">load_check_points</font></td> 
        <td>option for loading checkpoints model to train or test</td>
    </tr>
    <tr>
        <td><font color="green">batch_size</font></td> 
        <td>-</td>
    </tr>
    <tr>
        <td><font color="green">train_epochs</font></td> 
        <td>-</td>
    </tr>
    <tr>
        <td><font color="blue">shuffle</font></td> 
        <td>Option of the training data loader for shuffling data</td>
    </tr>
    <tr>
        <td><font color="green">num_workers</font></td> 
        <td>The number of workers when calling training data loader</td>
    </tr>
    <tr>
        <td><font color="green">drop_last</font></td> 
        <td>Option of the training data loader for droping last batch</td>
    </tr>
<tr>
        <td rowspan="3">Early-Stop Strategy</td>    
  		 <td><font color="blue">early_stop</font></td> 
      	 <td>option for employing early_stop</td>
    </tr>
    <tr>
        <td><font color="green">patience<font/></td> 
        <td>The times of tolerating performance degradation in valid dataset</td>
    </tr>
    <tr>
        <td><font color="blue">verbose<font/></td> 
        <td>Option for displaying the saving records of models</td>
    </tr>
<tr>
        <td rowspan="6">Optimizers</td>    
  		 <td><font color="green">optimizer<font/></td> 
      	 <td>Option for selecting optimizer, options:[Adam, AdamW, SGD, RMSprop,Adagrad, Adadelta]</td>
    </tr>
    <tr>
        <td><font color="green">lr<font/></td> 
        <td>learning rate</td>
    </tr>
    <tr>
        <td><font color="green">loss<font/></td> 
        <td>Option for selecting loss, options:mae, mas]</td>
    </tr>
    <tr>
        <td><font color="blue">smo_loss<font/></td> 
        <td>Option for constraint smoothing filter attention mechanism</td>
    </tr>
    <tr>
        <td><font color="blue">dynamic<font/></td> 
        <td>Option for ignoring larger loss in loss calculation</td>
    </tr>
    <tr>
        <td><font color="green">dynamic_ratio<font/></td> 
        <td>Ratio to filter out the larger data in a series sorted with an ascending order</td>
    </tr>
<tr>
        <td rowspan="6">Devices</td>    
  		 <td><font color="blue">use_gpu</font></td> 
      	 <td>Option for using GPU to train data</td>
    </tr>
    <tr>
        <td><font color="green">gpu_id</font></td> 
        <td>Option for employed GPU index</td>
    </tr>
    <tr>
        <td><font color="blue">use_multi_gpu</font></td> 
        <td>Option for employed multi GPUs</td>
    </tr>
    <tr>
        <td><font color="green">devices</font></td> 
        <td>The device ids of multi GPUs, default:[1,2,3,4]</td>
    </tr>
</table>

## Results
<p align="center">
<img src=".\.img/wthv.gif" height = "240" alt="" align=center />
<img src=".\.img/wth_3_.gif" height = "240" alt="" align=center />
<br><br>
<b>Figure 3.</b> An illustration of WTH prediction.
</p>
<p align="center">
<img src=".\.img/network_port8_1_.gif" height = "240" alt="" align=center />
<img src=".\.img/network_port9_1_.gif" height = "240" alt="" align=center />
<br><br>
<b>Figure 4.</b> An illustration of NPT prediction.
</p>




## Contact
If you have any questions, feel free to contact YuGuang Yang through Email (moujieguang@gmail.com) or Github issues. Pull requests are highly welcomed!

## Acknowlegements
![China Unicom](https://img.shields.io/badge/China%20Unicom-CC_BY--NC--SA--red.svg?color=critical) 
![Informer](https://img.shields.io/badge/Informer-CC_BY--NC--SA.svg?color=yellow)

Thanks for the data collected by China United Network Communications Group Co., Ltd. And the code organization framework is refferd from [Informer](https://github.com/zhouhaoyi/Informer2020).
At the same time, thank you all for your attention to this work! 
