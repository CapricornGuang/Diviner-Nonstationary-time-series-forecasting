python -u main.py --model=diviner --data=ETTh1 --predict_length=48 --enc_seq_len=7 --out_seq_len=2 --dec_seq_len=4 --dim_val=24 --dim_attn=12 --dim_attn_channel=48 --n_heads=6 --n_encoder_layers=3 --n_decoder_layers=2 --batch_size=32 --train_epochs=100 --use_gpu --smo_loss --dynamic --early_stop --shuffle --verbose --out_scale