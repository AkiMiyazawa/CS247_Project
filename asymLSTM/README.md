# AsymLSTM + DepthwiseMaxPool
Code was developed on Windows 10, but should be compatible with Linux systems. 

## Requirements:
```
torch >= 0.4.1
torchtext >= 0.1.1
numpy >= 1.13.1
scipy >= 0.19.0
sklearn >= 0.18.1
```

## Steps to reproduce AsymLSTM + DepthwiseMaxPool
1. Create a folder ./data/stackoverflow/ from this directory to store the unzipped .pt files downloaded from [here](https://drive.google.com/open?id=1Nm3OMbC24Z2HQyje0Xn2m23tW34y0G3_). 

2. Run the following command: 
```
python train.py -data_path_prefix "data/stackoverflow/stackoverflow" \
                -vocab_path "data/stackoverflow/stackoverflow.vocab.pt" \ 
                -exp_path "exp/asymm.uni.cellMax.lstm.concat.dropout.4enc1dec.512x150/%s.%s" \
                -exp "stackoverflow" \
                -copy_attention -copy_mode "concat" \
                -attention_mode "concat" -target_attention_mode "concat" \
                -input_feeding -copy_input_feeding \
                -train_ml -must_appear_in_src -dropout 0.2\
                -batch_workers 0 -batch_size 64 -beam_size 16 -beam_search_batch_size 16 \
                -rnn_size 512 -word_vec_size 150 \
                -enc_layers 4 -enc_cell_state_method 2 \
                -run_valid_every 1000 -save_model_every 2500
```
For additional or different configurations, please refer to .config.py

3. Training logs and test/validation results will be dumped into the .exp/ folder, which will be created automatically the first time you run the above command.

4. Feel free to email kaileymon@gmail.com if you have questions, thanks! 