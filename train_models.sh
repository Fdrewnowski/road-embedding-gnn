# download data
python ./download_graphs.py -t 
python ./download_graphs.py -v
# transform data and transform into dgl format
python transform_graphs.py -a -p ./data/data_train -o ./data/data_train/training_graphs.bin
python transform_graphs.py -a -p ./data/data_val -o ./data/data_val/validation_graphs.bin

python ./gae_train.py --use_cfg --full_pipline
python ./graphmae_train.py --use_cfg --full_pipline
python ./dgi_train.py --use_cfg --full_pipline
